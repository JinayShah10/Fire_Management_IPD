import argparse
import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import cv2
cv2.setNumThreads(0) # CRITICAL FIX: Prevent macOS C-level segmentation faults when using OpenCV in a ThreadPool
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import rasterio.mask
import requests
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("rasterio").setLevel(logging.WARNING)

def fully_covers(items, bounds):
    if not items:
        return False
    event_geom = box(*bounds)
    item_geoms = [box(*item['bbox']) for item in items]
    union_geom = unary_union(item_geoms)
    return union_geom.contains(event_geom)

def intersects(bbox1, bbox2):
    b1 = box(*bbox1)
    b2 = box(*bbox2)
    return b1.intersects(b2)

def load_stac_cache(metadata_path):
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load STAC cache: {e}")
    return {"io-lulc-9-class": [], "esa-worldcover": []}

def save_stac_cache(metadata_path, cache_dict):
    try:
        with open(metadata_path, 'w') as f:
            json.dump(cache_dict, f)
    except Exception as e:
        logger.warning(f"Failed to save STAC cache: {e}")

def validate_raster(path):
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 1024:
        return False
    try:
        with rasterio.open(path) as src:
            _ = src.profile
            return True
    except:
        return False

def download_asset(url, out_path):
    if validate_raster(out_path):
        return True, True  # success, cache_hit
        
    # Download
    retries = 3
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(out_path, 'wb') as f:
                    f.writelines(r.iter_content(chunk_size=8192))
            if validate_raster(out_path):
                return True, False
            else:
                os.remove(out_path)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Failed to download {url}: {e}")
                if os.path.exists(out_path):
                    os.remove(out_path)
    return False, False

transformers = {}
def get_transformer(from_crs, to_crs):
    key = (str(from_crs), str(to_crs))
    if key not in transformers:
        transformers[key] = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    return transformers[key]

def process_group(group_idx, group_df, args, lulc_paths, wc_paths, group_bbox):
    results = []
    
    if not lulc_paths or not wc_paths:
        for _, row in group_df.iterrows():
            d = row.to_dict()
            d['urban_filter_pass'] = False
            d['urban_filter_reason'] = "Missing required rasters"
            results.append(d)
        return results

    lulc_datasets = []
    wc_datasets = []
    try:
        try:
            lulc_datasets = [rasterio.open(p) for p in lulc_paths if os.path.exists(p)]
            wc_datasets = [rasterio.open(p) for p in wc_paths if os.path.exists(p)]
        except Exception as e:
            for _, row in group_df.iterrows():
                d = row.to_dict()
                d['urban_filter_pass'] = False
                d['urban_filter_reason'] = f"Raster error: {e}"
                results.append(d)
            return results
        
        for _, row in group_df.iterrows():
            event_dict = row.to_dict()
            try:
                lat = row['latitude']
                lon = row['longitude']
                
                deg_lat_m = 111320
                deg_lon_m = 111320 * np.cos(np.radians(lat))
                
                # --- LULC (Esri 10m) ---
                radius_m_lulc = args.ghsl_radius_km * 1000
                built_pixels_lulc = 0
                total_pixels_lulc = 0
                
                for src in lulc_datasets:
                    transformer = get_transformer("EPSG:4326", src.crs)
                    cx, cy = transformer.transform(lon, lat)
                    
                    if src.crs.is_geographic:
                        rx = radius_m_lulc / deg_lon_m
                        ry = radius_m_lulc / deg_lat_m
                    else:
                        rx, ry = radius_m_lulc, radius_m_lulc
                        
                    rmin_x, rmax_x = cx - rx, cx + rx
                    rmin_y, rmax_y = cy - ry, cy + ry
                    
                    if rmax_x < src.bounds.left or rmin_x > src.bounds.right or rmax_y < src.bounds.bottom or rmin_y > src.bounds.top:
                        continue
                        
                    window = rasterio.windows.from_bounds(rmin_x, rmin_y, rmax_x, rmax_y, transform=src.transform)
                    window_intersection = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                    window_intersection = window_intersection.round_lengths().round_offsets()
                    
                    if window_intersection.width <= 0 or window_intersection.height <= 0:
                        continue
                        
                    local_lulc = src.read(1, window=window_intersection)
                    built_pixels_lulc += np.sum(local_lulc == 7)
                    total_pixels_lulc += np.sum(local_lulc != 0)
                
                lulc_ratio = built_pixels_lulc / total_pixels_lulc if total_pixels_lulc > 0 else 0
                
                event_dict['ghsl_urban_ratio'] = lulc_ratio
                event_dict['ghsl_pass'] = lulc_ratio >= args.min_ghsl_urban_ratio
                if not event_dict['ghsl_pass']:
                    event_dict['urban_filter_pass'] = False
                    event_dict['urban_filter_reason'] = "GHSL (LULC) urban extent insufficient"
                    results.append(event_dict)
                    continue
                    
                # --- WorldCover ---
                radius_m_wc = args.worldcover_radius_km * 1000
                built_pixels_wc = 0
                total_pixels_wc = 0
                components = []
                
                for src in wc_datasets:
                    transformer = get_transformer("EPSG:4326", src.crs)
                    cx, cy = transformer.transform(lon, lat)
                    
                    res_x, res_y = abs(src.transform.a), abs(src.transform.e)
                    
                    if src.crs.is_geographic:
                        rx = radius_m_wc / deg_lon_m
                        ry = radius_m_wc / deg_lat_m
                        pixel_area_m2 = (res_x * deg_lon_m) * (res_y * deg_lat_m)
                        pixel_size_m = ((res_x * deg_lon_m) + (res_y * deg_lat_m)) / 2
                    else:
                        rx, ry = radius_m_wc, radius_m_wc
                        pixel_area_m2 = res_x * res_y
                        pixel_size_m = (res_x + res_y) / 2
                        
                    rmin_x, rmax_x = cx - rx, cx + rx
                    rmin_y, rmax_y = cy - ry, cy + ry
                    
                    if rmax_x < src.bounds.left or rmin_x > src.bounds.right or rmax_y < src.bounds.bottom or rmin_y > src.bounds.top:
                        continue
                        
                    window = rasterio.windows.from_bounds(rmin_x, rmin_y, rmax_x, rmax_y, transform=src.transform)
                    window_intersection = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                    window_intersection = window_intersection.round_lengths().round_offsets()
                    
                    if window_intersection.width <= 0 or window_intersection.height <= 0:
                        continue
                        
                    local_wc = src.read(1, window=window_intersection)
                    built_mask = (local_wc == 50).astype(np.uint8)
                    
                    built_pixels_wc += np.sum(built_mask)
                    total_pixels_wc += np.sum(local_wc != 0)
                    
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(built_mask, connectivity=8)
                    min_pixels = args.min_connected_area_m2 / pixel_area_m2
                    
                    fire_col, fire_row = ~src.transform * (cx, cy)
                    local_fire_c = int(fire_col - window_intersection.col_off)
                    local_fire_r = int(fire_row - window_intersection.row_off)
                    
                    for i in range(1, num_labels):
                        area = stats[i, cv2.CC_STAT_AREA]
                        if area >= min_pixels:
                            if 0 <= local_fire_r < local_wc.shape[0] and 0 <= local_fire_c < local_wc.shape[1]:
                                comp_mask = (labels == i).astype(np.uint8)
                                dist_transform = cv2.distanceTransform(1 - comp_mask, cv2.DIST_L2, 5)
                                dist_pixels = dist_transform[local_fire_r, local_fire_c]
                            else:
                                cx_pixel, cy_pixel = centroids[i]
                                dist_pixels = np.sqrt((cx_pixel - local_fire_c)**2 + (cy_pixel - local_fire_r)**2)
                                
                            dist_m = dist_pixels * pixel_size_m
                            components.append({'area_m2': area * pixel_area_m2, 'dist_m': dist_m})
                
                wc_ratio = built_pixels_wc / total_pixels_wc if total_pixels_wc > 0 else 0
                event_dict['worldcover_built_up_ratio'] = wc_ratio
                event_dict['worldcover_pass'] = wc_ratio >= args.min_built_up_ratio
                
                if not event_dict['worldcover_pass']:
                    event_dict['urban_filter_pass'] = False
                    event_dict['urban_filter_reason'] = "WorldCover built-up density too low"
                    results.append(event_dict)
                    continue
                    
                if not components:
                    event_dict['urban_filter_pass'] = False
                    event_dict['urban_filter_reason'] = "No sufficiently large connected built-up area"
                    event_dict['largest_builtup_component_area_m2'] = 0
                    results.append(event_dict)
                    continue
                    
                max_area = max(c['area_m2'] for c in components)
                min_dist = min(c['dist_m'] for c in components)
                
                event_dict['largest_builtup_component_area_m2'] = max_area
                event_dict['distance_to_builtup_m'] = min_dist
                
                if min_dist > args.max_distance_to_builtup_m:
                    event_dict['urban_filter_pass'] = False
                    event_dict['urban_filter_reason'] = "Fire too far from urban built-up area"
                    results.append(event_dict)
                    continue
                    
                event_dict['urban_filter_pass'] = True
                event_dict['urban_filter_reason'] = "PASSED"
                results.append(event_dict)
                
            except Exception as e:
                # Per event try/except avoids taking down the whole group
                event_dict['urban_filter_pass'] = False
                event_dict['urban_filter_reason'] = f"Event processing error: {e}"
                results.append(event_dict)

    finally:
        for src in lulc_datasets + wc_datasets:
            src.close()
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="stage_010_data_ingestion/data/events.csv")
    parser.add_argument("--out", default="stage_015_urban_filtering/data/events_urban.csv")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--download-workers", type=int, default=3)
    parser.add_argument("--ghsl-radius-km", type=float, default=2.0)
    parser.add_argument("--worldcover-radius-km", type=float, default=1.0)
    parser.add_argument("--min-ghsl-urban-ratio", type=float, default=0.20)
    parser.add_argument("--min-built-up-ratio", type=float, default=0.30)
    parser.add_argument("--min-connected-area-m2", type=float, default=100000)
    parser.add_argument("--max-distance-to-builtup-m", type=float, default=300)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    cache_dir = "stage_015_urban_filtering/cache"
    if args.clear_cache and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info("Cleared cache directory.")
        
    os.makedirs(f"{cache_dir}/rasters", exist_ok=True)
    os.makedirs(f"{cache_dir}/metadata", exist_ok=True)
    
    stac_metadata_path = f"{cache_dir}/metadata/stac_cache.json"
    stac_cache = load_stac_cache(stac_metadata_path)
    
    if not os.path.exists(args.events):
        logger.error(f"Input events not found: {args.events}")
        return
        
    df = pd.read_csv(args.events)
    logger.info(f"Loaded {len(df)} total raw events.")
    
    start_time = time.time()
    
    # 1. Group events spatially
    logger.info("[1/3] Grouping events and discovering required STAC assets...")
    df['grid_lon'] = np.floor(df['longitude'] / 0.1) * 0.1
    df['grid_lat'] = np.floor(df['latitude'] / 0.1) * 0.1
    
    # margin based on max radius (approx 0.02 degrees)
    margin = 0.02 
    
    groups = list(df.groupby(['grid_lat', 'grid_lon']))
    logger.info(f"Created {len(groups)} unique geographic groups.")
    
    catalog = None
    
    required_downloads = {} # path -> url
    group_assets = {} # group_idx -> {'lulc': [paths], 'wc': [paths], 'bbox': []}
    
    stac_searches_performed = 0
    stac_cache_hits = 0
    
    for idx, (group_key, group_df) in tqdm(enumerate(groups), total=len(groups), desc="STAC Discovery"):
        min_lon, max_lon = group_df['longitude'].min(), group_df['longitude'].max()
        min_lat, max_lat = group_df['latitude'].min(), group_df['latitude'].max()
        group_bbox = [min_lon - margin, min_lat - margin, max_lon + margin, max_lat + margin]
        
        group_assets[idx] = {'lulc': set(), 'wc': set(), 'bbox': group_bbox, 'df': group_df}
        
        for collection, asset_key in [("io-lulc-9-class", "data"), ("esa-worldcover", "map")]:
            # check cache first
            intersecting_cached = [i for i in stac_cache[collection] if intersects(i['bbox'], group_bbox)]
            if fully_covers(intersecting_cached, group_bbox):
                stac_cache_hits += 1
                items_to_use = intersecting_cached
            else:
                if catalog is None:
                    catalog = pystac_client.Client.open(
                        "https://planetarycomputer.microsoft.com/api/stac/v1"
                    )
                stac_searches_performed += 1
                search = catalog.search(collections=[collection], bbox=group_bbox)
                new_items = list(search.items())
                items_to_use = []
                for item in new_items:
                    item_dict = {
                        "id": item.id,
                        "bbox": item.bbox,
                        "href": item.assets[asset_key].href
                    }
                    if not any(k['id'] == item.id for k in stac_cache[collection]):
                        stac_cache[collection].append(item_dict)
                    items_to_use.append(item_dict)
            
            for item in items_to_use:
                if intersects(item['bbox'], group_bbox):
                    path = f"{cache_dir}/rasters/{collection}_{item['id']}_{asset_key}.tif"
                    required_downloads[path] = planetary_computer.sign_url(item['href'])
                    if collection == "io-lulc-9-class":
                        group_assets[idx]['lulc'].add(path)
                    else:
                        group_assets[idx]['wc'].add(path)
                        
    save_stac_cache(stac_metadata_path, stac_cache)
    
    logger.info(f"STAC Discovery complete. Cache hits: {stac_cache_hits}, API searches: {stac_searches_performed}.")
    logger.info(f"[2/3] Downloading/caching {len(required_downloads)} unique raster assets...")
    
    raster_downloads = 0
    raster_cache_hits = 0
    
    # We use ThreadPoolExecutor for downloading
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        futures = {executor.submit(download_asset, url, path): path for path, url in required_downloads.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Raster Assets"):
            success, cache_hit = future.result()
            if cache_hit:
                raster_cache_hits += 1
            elif success:
                raster_downloads += 1
                
    logger.info(f"Raster sync complete. Cache hits: {raster_cache_hits}, Downloads: {raster_downloads}.")
    
    # 3. Local processing
    logger.info(f"[3/3] Processing {len(df)} events locally in {len(groups)} blocks...")
    
    local_start_time = time.time()
    
    all_results = []
    
    import concurrent.futures
    # ProcessPoolExecutor for true multi-core speed! OpenCV segfault is fixed, so this is now safe.
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = set()
        progress = tqdm(total=len(group_assets), desc="Local Processing")
        keys = list(group_assets.keys())
        
        for k in keys:
            group_data = group_assets.pop(k) # Free memory dynamically!
            futures.add(executor.submit(process_group, k, group_data['df'], args, list(group_data['lulc']), list(group_data['wc']), group_data['bbox']))
            
            while len(futures) >= args.workers * 4:
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    all_results.extend(f.result())
                    progress.update(1)
                    
        for f in as_completed(futures):
            all_results.extend(f.result())
            progress.update(1)
            
        progress.close()
            
    local_end_time = time.time()
    
    out_df = pd.DataFrame(all_results)
    
    final_df = out_df[out_df['urban_filter_pass'] == True].copy()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    final_df.to_csv(args.out, index=False)
    
    diag_path = args.out.replace(".csv", "_diagnostic.csv")
    out_df.to_csv(diag_path, index=False)
    
    if len(final_df) > 0:
        logger.info(f"Wrote {len(final_df)} urban events to {args.out}")
        logger.info(f"Wrote diagnostic info to {diag_path}")
    else:
        logger.warning(f"No urban events found! Wrote empty file to {args.out}")
        
    total_time = time.time() - start_time
    
    logger.info("\n" + "="*50)
    logger.info("CACHE STATISTICS & BENCHMARK")
    logger.info("="*50)
    logger.info(f"Total events: {len(df)}")
    logger.info(f"Unique geographic groups: {len(groups)}")
    unique_esri = len([k for k in required_downloads if 'io-lulc' in k])
    unique_wc = len([k for k in required_downloads if 'esa-worldcover' in k])
    logger.info(f"Unique Esri assets: {unique_esri}")
    logger.info(f"Unique WorldCover assets: {unique_wc}")
    logger.info(f"STAC requests: {stac_searches_performed}")
    logger.info(f"STAC cache hits: {stac_cache_hits}")
    logger.info(f"Raster downloads: {raster_downloads}")
    logger.info(f"Raster cache hits: {raster_cache_hits}")
    logger.info(f"Total runtime: {total_time:.2f} s")
    logger.info(f"Local processing runtime: {local_end_time - local_start_time:.2f} s")
    logger.info(f"Network/download runtime: {local_start_time - start_time:.2f} s")
    events_per_sec = len(df) / (local_end_time - local_start_time) if (local_end_time - local_start_time) > 0 else 0
    logger.info(f"Events/second during local processing: {events_per_sec:.2f}")
    network_avoided = (len(df) * 2) - stac_searches_performed
    logger.info(f"Network STAC requests avoided: approximately {network_avoided}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
