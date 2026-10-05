import argparse
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

import yaml

def load_config():
    with open("configs/firms.yaml", "r") as f:
        firms_cfg = yaml.safe_load(f)
    return firms_cfg, None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--bbox", type=str, help="W,S,E,N")
    parser.add_argument("--sources", nargs='+')
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 # radius of earth in m
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    res = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)

def cluster_detections(df, dist_m):
    # O(N) grid-based clustering
    if len(df) == 0:
        return []
    
    events = []
    df = df.copy()
    
    # Grid of ~1.1km (0.01 degrees)
    df['cluster_lat'] = (df['latitude'] / 0.01).round() * 0.01
    df['cluster_lon'] = (df['longitude'] / 0.01).round() * 0.01
    
    groups = df.groupby(['acq_date', 'cluster_lat', 'cluster_lon'])
    for (date, clat, clon), cluster_group in tqdm(groups, total=len(groups), desc="Clustering Fires"):
        centroid_lat = cluster_group['latitude'].mean()
        centroid_lon = cluster_group['longitude'].mean()
        
        # deterministic hash of rounded centroid + date
        hash_str = f"{round(centroid_lat, 2)}_{round(centroid_lon, 2)}_{date}"
        event_id = "event_" + hashlib.md5(hash_str.encode()).hexdigest()[:8]
            
        times = cluster_group['acq_datetime']
        frp = cluster_group['frp']
        
        events.append({
            'event_id': event_id,
            'latitude': centroid_lat,
            'longitude': centroid_lon,
            'acq_date': date,
            'earliest_time': times.min().strftime("%H:%M"),
            'latest_time': times.max().strftime("%H:%M"),
            'n_detections': len(cluster_group),
            'max_frp': frp.max() if pd.notnull(frp).any() else 0,
            'sum_frp': frp.sum() if pd.notnull(frp).any() else 0,
            'sources': ",".join(sorted(cluster_group['source'].unique())),
            'satellites': ",".join(sorted(cluster_group['satellite'].unique())),
            'bbox': f"{cluster_group['longitude'].min()},{cluster_group['latitude'].min()},{cluster_group['longitude'].max()},{cluster_group['latitude'].max()}",
            'region': cluster_group['region'].iloc[0] if 'region' in cluster_group else '',
            'firms_source': cluster_group['source'].iloc[0] if 'source' in cluster_group else '',
            'firms_fetch_utc': cluster_group['firms_fetch_utc'].iloc[0] if 'firms_fetch_utc' in cluster_group else '',
            'firms_request_id': cluster_group['firms_request_id'].iloc[0] if 'firms_request_id' in cluster_group else '',
            'persistent_source': bool(cluster_group['persistent_source'].any()) if 'persistent_source' in cluster_group else False,
        })
            
    return events

def main():
    firms_cfg, _ = load_config()
    
    upload_dir = "stage_010_data_ingestion/data/user_uploaded_firms"
    if not os.path.exists(upload_dir):
        logger.error(f"Upload directory {upload_dir} not found.")
        return
        
    csv_files = [f for f in os.listdir(upload_dir) if f.endswith('.csv')]
    if not csv_files:
        logger.error("No CSV files found in upload directory. Please upload NASA bulk CSV files via the frontend.")
        return
        
    all_data = []
    for f in csv_files:
        path = os.path.join(upload_dir, f)
        logger.info(f"Reading uploaded file: {f}")
        try:
            df = pd.read_csv(path)
            # Add required internal columns
            if 'confidence' in df.columns:
                def norm_conf(c):
                    if pd.isnull(c): return 'u'
                    c_str = str(c).lower()
                    if c_str in ['l', 'low']: return 'low'
                    if c_str in ['n', 'nominal']: return 'nominal'
                    if c_str in ['h', 'high']: return 'high'
                    try:
                        num = float(c)
                        if num < 30: return 'low'
                        elif num < 80: return 'nominal'
                        else: return 'high'
                    except:
                        return c_str
                df['conf_norm'] = df['confidence'].apply(norm_conf)
            else:
                df['conf_norm'] = 'u'
                
            # Map correct source from instrument and satellite
            if 'instrument' in df.columns and 'satellite' in df.columns:
                df['source'] = df['instrument'].astype(str) + '_' + df['satellite'].astype(str)
            elif 'instrument' in df.columns:
                df['source'] = df['instrument']
            else:
                df['source'] = 'BULK_ARCHIVE'
                
            df['firms_request_id'] = f"file_{hashlib.md5(f.encode()).hexdigest()[:8]}"
            df['firms_fetch_utc'] = datetime.now(timezone.utc).isoformat()
            df['region'] = 'user_uploaded'
            all_data.append(df)
        except Exception as e:
            logger.error(f"Failed to read {f}: {e}")
            
    if not all_data:
        logger.info("No data fetched.")
        return
        
    full_df = pd.concat(all_data, ignore_index=True)
    raw_count = len(full_df)
    
    # parse datetime (Vectorized)
    t_str = full_df['acq_time'].astype(str).str.zfill(4)
    t_str = t_str.str[:2] + ':' + t_str.str[2:]
    # Replace :: with : just in case some already had colons
    t_str = t_str.str.replace("::", ":")
    full_df['acq_datetime'] = pd.to_datetime(full_df['acq_date'].astype(str) + ' ' + t_str, utc=True)
    
    # drop duplicates across same source/coords/time
    full_df = full_df.drop_duplicates(subset=['source', 'latitude', 'longitude', 'acq_datetime'])
    dedup_count = len(full_df)
    
    # Filtering
    filters = firms_cfg['filters']
    
    # daytime only
    f_day = full_df[full_df['daynight'].str.upper() == filters['daynight']]
    day_count = len(f_day)
    
    # confidence
    valid_conf = [c.lower() for c in filters['confidence']]
    f_conf = f_day[f_day['conf_norm'].isin(valid_conf) | f_day['confidence'].isin(valid_conf)]
    conf_count = len(f_conf)
    
    # FRP
    f_frp = f_conf[f_conf['frp'] >= filters['min_frp']]
    frp_count = len(f_frp)
    
    final_df = f_frp.copy()
    
    # Persistent heat flag
    # 0.01 deg grid
    full_df['grid_lat'] = (full_df['latitude'] / 0.01).round() * 0.01
    full_df['grid_lon'] = (full_df['longitude'] / 0.01).round() * 0.01
    grid_counts = full_df.groupby(['grid_lat', 'grid_lon'])['acq_date'].nunique()
    persistent_grids = grid_counts[grid_counts > firms_cfg['persistent_days']].index
    
    final_df['grid_lat'] = (final_df['latitude'] / 0.01).round() * 0.01
    final_df['grid_lon'] = (final_df['longitude'] / 0.01).round() * 0.01
    persistent_set = set(persistent_grids)
    final_df['persistent_source'] = [ (lat, lon) in persistent_set for lat, lon in zip(final_df['grid_lat'], final_df['grid_lon']) ]
    
    events = cluster_detections(final_df, firms_cfg['cluster_dist_m'])
    events_df = pd.DataFrame(events)
    
    if len(events_df) > 0:
        events_df.to_csv("stage_010_data_ingestion/data/events.csv", index=False)
    
    
    report = {
        'requests_made': 0,
        'cache_hits': 0,
        'transactions_before': 0,
        'transactions_after': 0,
        'funnel': {
            'raw': raw_count,
            'dedup': dedup_count,
            'daytime': day_count,
            'confidence': conf_count,
            'frp': frp_count,
            'events': len(events)
        }
    }
    with open("stage_010_data_ingestion/data/fetch_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Fetch complete. Wrote {len(events)} events.")
    logger.info(json.dumps(report['funnel'], indent=2))

if __name__ == "__main__":
    main()
