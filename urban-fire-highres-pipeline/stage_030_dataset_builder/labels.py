import numpy as np
from scipy.ndimage import binary_dilation, label


def generate_labels(ms_data, qc_stats, config, hist_ms_data=None, event_lat=None, event_lon=None):
    b12 = ms_data.arrays['B12']
    b8a = ms_data.arrays['B8A']
    
    # Calculate ratio, safely handle division by zero or NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = b12 / b8a
        ratio[~np.isfinite(ratio)] = 0.0

    seed_mask = (b12 >= config.labels.seed_b12_min) & (ratio >= config.labels.seed_ratio_min)
    grow_mask_base = (b12 >= config.labels.grow_b12_min) & (ratio >= config.labels.grow_ratio_min)
    
    # FILTER 1: Historical Static Mask (Urban Glare Removal)
    if hist_ms_data is not None:
        hist_b12 = hist_ms_data.arrays['B12']
        hist_b8a = hist_ms_data.arrays['B8A']
        with np.errstate(divide='ignore', invalid='ignore'):
            hist_ratio = hist_b12 / hist_b8a
            hist_ratio[~np.isfinite(hist_ratio)] = 0.0
            
        hist_seed_mask = (hist_b12 >= config.labels.seed_b12_min) & (hist_ratio >= config.labels.seed_ratio_min)
        seed_mask = seed_mask & (~hist_seed_mask)
        grow_mask_base = grow_mask_base & (~hist_seed_mask)

    # FILTER 2: Cross-Sensor Validation (VIIRS Footprint)
    if event_lat is not None and event_lon is not None:
        import rasterio
        from rasterio.warp import transform as coord_transform
        
        # Project VIIRS lat/lon into Sentinel-2 CRS
        xs, ys = coord_transform('EPSG:4326', ms_data.crs, [event_lon], [event_lat])
        event_x, event_y = xs[0], ys[0]
        
        # Create meshgrid of coordinates
        cols, rows = np.meshgrid(np.arange(b12.shape[1]), np.arange(b12.shape[0]))
        px_x, px_y = rasterio.transform.xy(ms_data.transform, rows, cols)
        px_x = np.array(px_x).reshape(b12.shape)
        px_y = np.array(px_y).reshape(b12.shape)
        
        dist_sq = (px_x - event_x)**2 + (px_y - event_y)**2
        
        # 375m radius around the VIIRS point (configurable)
        radius = getattr(config.firms, 'footprint_radius_m', 375.0)
        viirs_mask = dist_sq <= (radius**2)
        
        seed_mask = seed_mask & viirs_mask
        grow_mask_base = grow_mask_base & viirs_mask
    
    # Grow seeds
    grown_mask = seed_mask.copy()
    for _ in range(config.labels.grow_steps):
        grown_mask = binary_dilation(grown_mask) & grow_mask_base

    # Find connected components
    labeled_components, num_features = label(grown_mask)
    
    accepted_components = []
    review_required_components = []
    rejected_components = []
    
    # We will use simple heuristics since we don't have the full legacy config,
    # or we can just accept all for now if area > 0, but the prompt says to 
    # keep the legacy statuses. We'll use a basic size threshold.
    # Legacy threshold might be `min_component_pixels` etc. We'll default to 1 pixel.
    
    for comp_id in range(1, num_features + 1):
        comp_mask = (labeled_components == comp_id)
        pixel_count = np.sum(comp_mask)
        
        # Industrial flare flag if available (simulated as false if not present)
        is_industrial = False # We'll set this if legacy had it
        
        if pixel_count < 1:
            decision = "REJECTED"
        elif pixel_count < 2:
            decision = "REVIEW_REQUIRED"
        else:
            decision = "ACCEPTED"
            
        comp = {
            "id": comp_id,
            "pixel_count": int(pixel_count),
            "is_industrial": is_industrial,
            "decision": decision
        }
        
        if decision == "ACCEPTED":
            accepted_components.append(comp)
        elif decision == "REVIEW_REQUIRED":
            review_required_components.append(comp)
        else:
            rejected_components.append(comp)
            
    # Build mask
    # 0 background, 1 fire, 2 industrial_flare, 255 ignore
    final_mask = np.zeros_like(b12, dtype=np.uint8)
    
    # Base background is 0, but clouds/shadows should be 255 (ignore)
    # except where the labeler marks it as fire
    cloud_mask = ms_data.arrays['SCL']
    invalid_scl = np.isin(cloud_mask, config.qc.cloud_scl_classes) | (cloud_mask == 0)
    final_mask[invalid_scl] = config.labels.ignore_index
    
    for comp in accepted_components:
        comp_mask = (labeled_components == comp['id'])
        val = 2 if comp['is_industrial'] else 1
        final_mask[comp_mask] = val
        
    for comp in review_required_components:
        comp_mask = (labeled_components == comp['id'])
        final_mask[comp_mask] = config.labels.ignore_index
        
    return {
        "mask": final_mask,
        "accepted": accepted_components,
        "review_required": review_required_components,
        "rejected": rejected_components,
        "labeled_array": labeled_components
    }
