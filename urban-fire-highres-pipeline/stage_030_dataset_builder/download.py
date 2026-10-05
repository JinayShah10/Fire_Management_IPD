import logging
import os
from dataclasses import dataclass
from datetime import timedelta, timezone

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import transform as coord_transform
from rasterio.windows import from_bounds

from shared.config import get_config

logger = logging.getLogger(__name__)

@dataclass
class MultispectralData:
    arrays: dict[str, np.ndarray]  # 'B12', 'B11', 'B8A', 'B8', 'B4', 'B3', 'B2', 'SCL'
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    scene_id: str
    acquisition_datetime: str
    processing_baseline: str
    offset_applied: float
    product_level: str
    time_offset_hours: float
    bounds: tuple

def get_catalog():
    cfg = get_config()
    return pystac_client.Client.open(
        cfg.stac.url
    )

def search_closest_scene(lat: float, lon: float, event_dt, config):
    if event_dt is None:
        return None, None
        
    # Widen search window to +/- 7 days so we can find a scene even if it's outside the threshold,
    # just so we can log the true closest offset in the quarantine reason.
    search_days = 7.0
    start_dt = event_dt - timedelta(days=search_days)
    end_dt = event_dt + timedelta(days=search_days)
    time_window = f"{start_dt.isoformat()}/{end_dt.isoformat()}"
    
    # 0.05 deg is approx 5km
    bbox = [lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05]
    
    try:
        catalog = get_catalog()
        search = catalog.search(
            collections=[config.stac.collection],
            bbox=bbox,
            datetime=time_window
        )
        items = list(search.items())
    except Exception as e:
        logger.error(f"STAC search failed: {e}")
        return None, None, f"STAC search failed: {e}"
        
    if not items:
        return None, None, "No items found in +/- 7 days"
        
    best_item = None
    min_time_diff = None
    
    logger.info(f"Found {len(items)} candidates for event at {event_dt.isoformat()}")
    
    same_day_candidates = []
    other_candidates = []
    
    for item in items:
        item_dt = item.datetime
        if not item_dt.tzinfo:
            item_dt = item_dt.replace(tzinfo=timezone.utc)
            
        time_diff = abs((item_dt - event_dt).total_seconds()) / 3600.0
        logger.info(f"Candidate: {item.id}, Offset: {time_diff:.2f} hours")
        
        if time_diff > config.firms.max_time_offset_hours:
            continue
            
        if getattr(config.firms, 'prefer_same_day', True) and item_dt.date() == event_dt.date():
            same_day_candidates.append((time_diff, item))
        else:
            other_candidates.append((time_diff, item))
            
    if same_day_candidates:
        same_day_candidates.sort(key=lambda x: x[0])
        best_item = same_day_candidates[0][1]
        min_time_diff = same_day_candidates[0][0]
    elif other_candidates:
        other_candidates.sort(key=lambda x: x[0])
        best_item = other_candidates[0][1]
        min_time_diff = other_candidates[0][0]
        
    if best_item:
        logger.info(f"Selected: {best_item.id} with offset {min_time_diff:.2f} hours")
        return best_item, min_time_diff, "Success"
        
    # If no valid items within max_time_offset_hours, find the absolute closest to report
    all_offsets = [abs((i.datetime.replace(tzinfo=timezone.utc) if not i.datetime.tzinfo else i.datetime) - event_dt).total_seconds() / 3600.0 for i in items]
    closest = min(all_offsets) if all_offsets else None
    
    return None, None, {"reason": f"Closest scene is {closest:.2f} hours away, which exceeds max {config.firms.max_time_offset_hours} hours"}

def search_historical_scene(lat: float, lon: float, event_dt, config):
    if event_dt is None:
        return None
        
    start_dt = event_dt - timedelta(days=config.reference.days_before_max)
    end_dt = event_dt - timedelta(days=config.reference.days_before_min)
    time_window = f"{start_dt.isoformat()}/{end_dt.isoformat()}"
    
    bbox = [lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05]
    
    try:
        catalog = get_catalog()
        search = catalog.search(
            collections=[config.stac.collection],
            bbox=bbox,
            datetime=time_window,
            query={"eo:cloud_cover": {"lt": 20}}  # Prefer low cloud cover
        )
        items = list(search.items())
    except Exception as e:
        logger.error(f"STAC historical search failed: {e}")
        return None
        
    if not items:
        return None
        
    # Sort by lowest cloud cover
    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
    return items[0]


def download_patch(item, lat: float, lon: float, time_offset_hours: float, config) -> MultispectralData | None:
    
    # Determine bounds from lat/lon and patch size
    target_res = config.grid.resolution_m
    master_px = 512  # Download 512x512 master so we can crop 256x256 patches with jitter
    half_size = (master_px * target_res) / 2.0
    
    env_kwargs = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "tif,tiff,jp2",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_HTTP_MULTIMAC": "YES",
        "GDAL_HTTP_MAX_RETRY": "5",
        "VSI_CACHE": "TRUE"
    }
    
    swir22_asset = getattr(config.stac.assets, "B12", "swir22")
    
    href_swir = item.assets[swir22_asset].href
    
    baseline = item.properties.get("s2:processing_baseline", "00.00")
    # Apply offset according to baseline rules
    offset_val = -1000.0 if float(baseline) >= 4.0 else 0.0
    logger.info(f"Using offset {offset_val} based on processing baseline {baseline}")
    
    arrays = {}
    master_transform = None
    master_shape = None
    crs = None
    
    with rasterio.Env(**env_kwargs):
        # 1. Establish grid using SWIR2 (native 20m)
        with rasterio.open(href_swir) as src:
            crs = src.crs
            xs, ys = coord_transform('EPSG:4326', crs, [lon], [lat])
            cx, cy = xs[0], ys[0]
            left, bottom, right, top = cx - half_size, cy - half_size, cx + half_size, cy + half_size
            
            # Master grid definitions
            master_transform = rasterio.transform.from_bounds(left, bottom, right, top, master_px, master_px)
            master_shape = (master_px, master_px)
            
            # Read SWIR22 (B12)
            window = from_bounds(left, bottom, right, top, src.transform).round_lengths().round_offsets()
            arr = src.read(
                window=window,
                out_shape=(1, master_px, master_px),
                resampling=rasterio.enums.Resampling.nearest,
                boundless=True,
                fill_value=0
            )
            raw = arr[0].astype(np.float32)
            valid = raw > 0
            refl = np.full_like(raw, np.nan)
            refl[valid] = np.clip(
                (raw[valid] + offset_val) / 10000.0, 
                config.normalization.reflectance_clip[0], 
                config.normalization.reflectance_clip[1]
            )
            arrays['B12'] = refl

        # Helper to fetch other bands
        def fetch_band(asset_key, native_res_m, band_name):
            asset_key_str = getattr(config.stac.assets, band_name, None)
            href = item.assets[asset_key_str].href
            with rasterio.open(href) as src:
                window = from_bounds(left, bottom, right, top, src.transform).round_lengths().round_offsets()
                if native_res_m < target_res and band_name != 'SCL':
                    resampling = rasterio.enums.Resampling.average
                else:
                    resampling = rasterio.enums.Resampling.nearest
                    
                arr = src.read(
                    window=window,
                    out_shape=(1, master_px, master_px),
                    resampling=resampling,
                    boundless=True,
                    fill_value=0
                )
                
                raw = arr[0].astype(np.float32)
                
                if band_name == 'SCL':
                    return band_name, raw
                
                valid = raw > 0
                refl = np.full_like(raw, np.nan)
                refl[valid] = np.clip(
                    (raw[valid] + offset_val) / 10000.0, 
                    config.normalization.reflectance_clip[0], 
                    config.normalization.reflectance_clip[1]
                )
                return band_name, refl

        # Fetch remaining bands synchronously or asynchronously
        # For simplicity and given rasterio limitations sometimes, we'll do sequential here 
        # (or concurrent if needed, but sequential is safer for VSI)
        bands_to_fetch = [
            (20, "B11"),
            (20, "B8A"),
            (10, "B8"),
            (10, "B4"),
            (10, "B3"),
            (10, "B2"),
            (20, "SCL")
        ]
        
        for native_res, b_name in bands_to_fetch:
            b, data = fetch_band(None, native_res, b_name)
            arrays[b] = data

    for name, arr in arrays.items():
        if arr.shape != master_shape:
            err_dict = {
                "error": "grid_mismatch",
                "band": name,
                "expected_shape": master_shape,
                "actual_shape": arr.shape,
                "event_id": f"{lat}_{lon}"
            }
            import json
            os.makedirs("stage_030_dataset_builder/data/quarantine/grid_mismatch", exist_ok=True)
            with open(f"stage_030_dataset_builder/data/quarantine/grid_mismatch/err_{item.id}_{name}.json", "w") as f:
                json.dump(err_dict, f)
            raise ValueError(f"Band {name} has incorrect shape {arr.shape} != {master_shape}")

    return MultispectralData(
        arrays=arrays,
        transform=master_transform,
        crs=crs,
        scene_id=item.id,
        acquisition_datetime=item.datetime.isoformat() if hasattr(item.datetime, 'isoformat') else str(item.datetime),
        processing_baseline=baseline,
        offset_applied=offset_val,
        product_level="Level-2A",
        time_offset_hours=time_offset_hours,
        bounds=(left, bottom, right, top)
    )
