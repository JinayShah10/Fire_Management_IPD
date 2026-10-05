import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, timezone
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from shared.config import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_catalog():
    import planetary_computer
    import pystac_client
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

def _search_stac(row, config, s2_prescreen, catalog):
    lat = row['latitude']
    lon = row['longitude']
    try:
        event_dt = pd.to_datetime(row['acq_date'] + " " + row['earliest_time']).replace(tzinfo=timezone.utc)
    except:
        event_dt = pd.to_datetime(row['acq_date'] + " " + row.get('acq_time', "12:00")).replace(tzinfo=timezone.utc)
        
    start_dt = event_dt - timedelta(days=1)
    end_dt = event_dt + timedelta(days=1)
    time_window = f"{start_dt.isoformat()}/{end_dt.isoformat()}"
    
    bbox = [lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001]
    
    retries = 3
    for attempt in range(retries):
        try:
            search = catalog.search(
                collections=[config.stac.collection],
                bbox=bbox,
                datetime=time_window,
                query={"eo:cloud_cover": {"lt": s2_prescreen.get('max_scene_cloud', 40)}}
            )
            items = list(search.items())
            break
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < retries - 1:
                time.sleep(5 + (2 ** attempt))
                continue
            logger.warning(f"STAC search failed for event {row['event_id']}: {e}")
            items = []
            break
        
    if not items:
        return None
        
    # Pick closest
    best_item = None
    min_diff = float('inf')
    
    for item in items:
        scene_dt = pd.to_datetime(item.datetime).replace(tzinfo=timezone.utc)
        diff_hours = abs((scene_dt - event_dt).total_seconds()) / 3600.0
        if diff_hours < min_diff:
            min_diff = diff_hours
            best_item = item
            
    row_dict = row.to_dict()
    row_dict['s2_scene_id'] = best_item.id
    row_dict['s2_cloud_cover'] = best_item.properties.get('eo:cloud_cover', -1)
    row_dict['s2_time_offset_hours'] = min_diff
    
    # drop date_obj and year_month
    row_dict.pop('date_obj', None)
    row_dict.pop('year_month', None)
    return row_dict

def run_prescreen(csv_path, out_csv, config):
    df = pd.read_csv(csv_path)
    
    with open("configs/firms.yaml", "r") as f:
        firms_cfg = yaml.safe_load(f)
        s2_prescreen = firms_cfg.get('s2_prescreen', {})
    
    # Check if region is present
    if 'region' not in df.columns:
        df['region'] = 'unknown'

    df['date_obj'] = pd.to_datetime(df['acq_date'])
    df['year_month'] = df['date_obj'].dt.strftime('%Y-%m')
    
    max_per = s2_prescreen.get('max_events_per_region_month', 200)
    capped_df = df.groupby(['region', 'year_month'], group_keys=False).apply(
        lambda x: x.sample(min(len(x), max_per), random_state=42)
    ).reset_index(drop=True)
    
    # Ensure region column exists just in case pandas dropped it
    if 'region' not in capped_df.columns:
        # Re-merge region from original df if it was lost
        capped_df = capped_df.merge(df[['event_id', 'region']], on='event_id', how='left')
    
    prescreened = []
    
    catalog = get_catalog() # safe to share thread-safe client for searches? Usually yes for read requests
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_search_stac, row, config, s2_prescreen, catalog) for idx, row in capped_df.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="STAC Prescreening"):
            res = future.result()
            if res is not None:
                prescreened.append(res)
                
    out_df = pd.DataFrame(prescreened) if prescreened else pd.DataFrame(columns=capped_df.columns if len(capped_df) > 0 else df.columns)
    out_df.to_csv(out_csv, index=False)
    
    if prescreened:
        logger.info(f"Wrote {len(out_df)} prescreened events to {out_csv}")
    else:
        logger.warning(f"No events passed prescreening. Wrote empty file to {out_csv}")
        
    return len(df), len(capped_df), len(prescreened), prescreened

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="stage_015_urban_filtering/data/events_urban.csv")
    parser.add_argument("--out", default="stage_020_prescreening/data/events_prescreened.csv")
    parser.add_argument("--report", default="stage_010_data_ingestion/data/fetch_report.json")
    args = parser.parse_args()
    
    config = get_config()
    
    if not os.path.exists(args.events):
        logger.error(f"Input events not found: {args.events}")
        return
        
    raw_events, capped_events, matched_events, prescreened_rows = run_prescreen(args.events, args.out, config)
    
    if os.path.exists(args.report):
        with open(args.report, "r") as f:
            report = json.load(f)
            
        report['funnel']['events_capped'] = capped_events
        report['funnel']['events_with_s2'] = matched_events
        
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
            
        print("=== FUNNEL TABLE ===")
        print(f"Raw detections:                 {report['funnel'].get('raw', 0)}")
        print(f"After daytime filter:           {report['funnel'].get('daytime', 0)}")
        print(f"After confidence filter:        {report['funnel'].get('confidence', 0)}")
        print(f"After FRP filter:               {report['funnel'].get('frp', 0)}")
        print(f"Events (clusters):              {report['funnel'].get('events', 0)}")
        print(f"Events (capped):                {capped_events}")
        print(f"Events with S2 scene (+/- 1d):  {matched_events}")
    
    if prescreened_rows:
        df = pd.DataFrame(prescreened_rows)
        
        plt.figure(figsize=(10, 6))
        for region, group in df.groupby('region'):
            plt.scatter(group['longitude'], group['latitude'], label=region, alpha=0.6)
        plt.legend()
        plt.title("FIRMS Events Map")
        plt.savefig("stage_020_prescreening/data/firms_events_map.png")
        plt.close()
        
        df['month'] = pd.to_datetime(df['acq_date']).dt.to_period('M')
        df['month'].value_counts().sort_index().plot(kind='bar')
        plt.title("Events per Month")
        plt.savefig("stage_020_prescreening/data/events_per_month.png")
        plt.close()

if __name__ == "__main__":
    main()
