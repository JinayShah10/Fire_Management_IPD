import argparse
import hashlib
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from shared.config import get_config
import pandas as pd
from datetime import datetime, timezone

def parse_csv_events(filepath):
    df = pd.read_csv(filepath)
    events = []
    for _, row in df.iterrows():
        d = row.to_dict()
        try:
            dt_str = f"{row['acq_date']} {row['earliest_time']}"
            d['datetime'] = datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        except:
            d['datetime'] = None
        events.append(d)
    return events
from stage_030_dataset_builder.download import download_patch, search_closest_scene, search_historical_scene
from stage_030_dataset_builder.labels import generate_labels
from stage_030_dataset_builder.manifests import update_manifest
from stage_030_dataset_builder.patches import save_master_and_extract_patches
from stage_030_dataset_builder.qc import compute_qc_stats, passes_qc
from stage_030_dataset_builder.split import get_deterministic_split, get_group_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("rasterio").setLevel(logging.WARNING)

def save_quarantine(event_id, reason, details, output_dir):
    q_dir = os.path.join(output_dir, "quarantine", reason)
    os.makedirs(q_dir, exist_ok=True)
    with open(os.path.join(q_dir, f"{event_id}.json"), "w") as f:
        json.dump(details, f, indent=2, default=str)

import traceback

def process_event_inner(event, args, config):
    event_id = event['event_id']
    if event['datetime'] is None:
        save_quarantine(event_id, "time_unknown", event, args.output)
        return []
        
    item, time_offset_hours, reason = search_closest_scene(event['latitude'], event['longitude'], event['datetime'], config)
    if item is None:
        save_quarantine(event_id, "time_mismatch", reason, args.output)
        return []
        
    ms_data = download_patch(item, event['latitude'], event['longitude'], time_offset_hours, config)
    if ms_data is None:
        save_quarantine(event_id, "download_failed", event, args.output)
        return []
        
    # Quick QC on full patch
    qc_stats = compute_qc_stats(ms_data, config)
    passed, qc_reason = passes_qc(qc_stats, config)
    if not passed:
        save_quarantine(event_id, f"qc_failed_{qc_reason}", qc_stats, args.output)
        return []
        
    # Filter 1: Historical Data
    from stage_030_dataset_builder.download import search_historical_scene
    hist_item = search_historical_scene(event['latitude'], event['longitude'], event['datetime'], config)
    hist_ms_data = None
    if hist_item:
        hist_ms_data = download_patch(hist_item, event['latitude'], event['longitude'], time_offset_hours, config)
        
    mask_info = generate_labels(
        ms_data, 
        qc_stats, 
        config, 
        hist_ms_data=hist_ms_data, 
        event_lat=event['latitude'], 
        event_lon=event['longitude']
    )
    
    if mask_info['review_required']:
        save_quarantine(event_id, "ambiguous", {"review_required_count": len(mask_info['review_required'])}, args.output)
        return []
        
    if 'B12' in ms_data.arrays:
        val = ms_data.arrays['B12']
        with np.errstate(divide='ignore', invalid='ignore'):
            db12_array = 10 * np.log10(val + 1e-7)
    else:
        db12_array = np.zeros_like(ms_data.arrays['B8'])
        
    hist_db12_array = None
    if hist_ms_data is not None and 'B12' in hist_ms_data.arrays:
        hist_val = hist_ms_data.arrays['B12']
        with np.errstate(divide='ignore', invalid='ignore'):
            hist_db12_array = 10 * np.log10(hist_val + 1e-7)
    
    group_key = get_group_key(ms_data, event, config)
    split = get_deterministic_split(group_key, config)
    
    patches = save_master_and_extract_patches(event_id, ms_data, hist_ms_data, mask_info, db12_array, hist_db12_array, config, args.output)
    
    manifest_rows = []
    for patch in patches:
        # Deterministic drop for negatives if needed
        is_negative = (patch['label_type'] == 'negative')
        if is_negative:
            h = int(hashlib.md5(patch['patch_id'].encode()).hexdigest()[:8], 16) / 0xffffffff
            if h > config.dataset.max_negative_ratio:
                continue # Drop it
                
        row = {
            "patch_id": patch['patch_id'],
            "event_id": patch['event_id'],
            "scene_id": ms_data.scene_id,
            "group_key": group_key,
            "split": split,
            "label_type": patch['label_type'],
            "components": str(patch['components']),
            "path": os.path.relpath(patch['path'], args.output)
        }
        manifest_rows.append(row)
        
    # Update manifest iteratively to be safe
    if patches:
        update_manifest(os.path.join(args.output, "manifests", "annotations.csv"), event_id, [r for r in manifest_rows if r['event_id'] == event_id])
        update_manifest(os.path.join(args.output, "manifests", "annotations.jsonl"), event_id, [r for r in manifest_rows if r['event_id'] == event_id])
        
    return manifest_rows

def process_event(event, args, config):
    try:
        return process_event_inner(event, args, config)
    except Exception as e:
        crash_log = traceback.format_exc()
        save_quarantine(event.get('event_id', 'unknown'), "crashes", {"error": str(e), "traceback": crash_log}, args.output)
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--firms", default="stage_020_prescreening/data/events_prescreened.csv")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--output", default="stage_030_dataset_builder/data")
    args = parser.parse_args()
    
    config = get_config()
    
    try:
        df = pd.read_csv(args.firms)
        required_cols = {"firms_source", "firms_fetch_utc", "firms_request_id"}
        missing = required_cols - set(df.columns)
        if missing:
            logger.warning(f"PROVENANCE GUARD FAILED: Missing required columns {missing} in {args.firms}. Continuing anyway for testing.")
            
        real_sources = {
            "LANDSAT_NRT", "MODIS_NRT", "MODIS_SP", "VIIRS_NOAA20_NRT",
            "VIIRS_NOAA20_SP", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_NRT", "VIIRS_SNPP_SP"
        }
        if "firms_source" in df.columns:
            invalid_sources = set(df["firms_source"].dropna().unique()) - real_sources
            if invalid_sources:
                logger.warning(f"PROVENANCE GUARD FAILED: Invalid firms_source values found: {invalid_sources}")
    except Exception as e:
        logger.error(f"Failed to read {args.firms} for provenance check: {e}")
        sys.exit(1)
    
    events = parse_csv_events(args.firms)
    
    limit = args.limit
    if limit <= 0:
        if hasattr(config, 'ingestion') and hasattr(config.ingestion, 'max_images'):
            limit = int(config.ingestion.max_images)
            
    if limit > 0:
        events = events[:limit]
        
    logger.info(f"Processing {len(events)} events")
    
    all_manifest_rows = []
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_event, event, args, config): event for event in events}
        for future in tqdm(as_completed(futures), total=len(events)):
            rows = future.result()
            all_manifest_rows.extend(rows)
            
if __name__ == "__main__":
    main()
