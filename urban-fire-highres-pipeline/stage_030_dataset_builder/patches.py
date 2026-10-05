import logging
import os
import random

import numpy as np

from stage_030_dataset_builder.qc import passes_qc

logger = logging.getLogger(__name__)

CHANNEL_ORDER = ['B12', 'B11', 'B8A', 'B8', 'B4', 'B3', 'B2', 'dB12']

def save_master_and_extract_patches(event_id, ms_data, hist_data, mask_info, db12_array, hist_db12_array, config, out_dir):
    """
    Saves a master NPZ for the event and crops out 256x256 patches.
    """
    os.makedirs(os.path.join(out_dir, "masters"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "patches"), exist_ok=True)
    
    # Assemble master 512x512 array
    # Shape: (16, 512, 512)
    arrays = []
    # Current image
    for ch in CHANNEL_ORDER:
        if ch == 'dB12':
            arrays.append(db12_array)
        else:
            arrays.append(ms_data.arrays[ch])
            
    # Historical image
    for ch in CHANNEL_ORDER:
        if ch == 'dB12':
            if hist_db12_array is not None:
                arrays.append(hist_db12_array)
            else:
                arrays.append(np.zeros_like(db12_array)) # Fallback if historical is missing
        else:
            if hist_data is not None:
                arrays.append(hist_data.arrays[ch])
            else:
                arrays.append(np.zeros_like(db12_array))
            
    master_stack = np.stack(arrays, axis=0).astype(np.float32)
    mask = mask_info['mask']
    
    # Save master
    master_path = os.path.join(out_dir, "masters", f"{event_id}_master.npz")
    np.savez_compressed(
        master_path,
        stack=master_stack,
        mask=mask,
        transform=np.array(ms_data.transform).astype(np.float64),
        scene_id=ms_data.scene_id,
        time_offset_hours=ms_data.time_offset_hours
    )
    
    patches_extracted = []
    
    patch_px = config.grid.patch_px
    master_px = master_stack.shape[1]
    
    # 1. Identify connected components (fire) for positive patches
    # If no accepted components -> negative
    accepted = mask_info['accepted']
    
    def extract_patch(y_start, x_start, label_type):
        y_end = y_start + patch_px
        x_end = x_start + patch_px
        
        patch_stack = master_stack[:, y_start:y_end, x_start:x_end]
        patch_mask = mask[y_start:y_end, x_start:x_end]
        
        # QC for patch
        patch_scl = ms_data.arrays['SCL'][y_start:y_end, x_start:x_end]
        patch_b12 = patch_stack[0] # B12 is at index 0
        
        total_pixels = patch_px * patch_px
        cloud_mask = np.isin(patch_scl, config.qc.cloud_scl_classes)
        cloud_frac = np.sum(cloud_mask) / total_pixels
        
        nodata_mask = np.isnan(patch_b12)
        nodata_frac = np.sum(nodata_mask) / total_pixels
        
        qc_stats = {
            "cloud_fraction": float(cloud_frac),
            "nodata_fraction": float(nodata_frac)
        }
        
        passed, reason = passes_qc(qc_stats, config)
        if not passed:
            return None, reason
            
        patch_id = f"{event_id}_{label_type}_{y_start}_{x_start}"
        patch_path = os.path.join(out_dir, "patches", f"{patch_id}.npz")
        
        # Determine specific component ids present in patch
        unique_labels = np.unique(mask_info['labeled_array'][y_start:y_end, x_start:x_end])
        comps_present = [int(u) for u in unique_labels if u > 0]
        
        np.savez_compressed(
            patch_path,
            stack=patch_stack,
            mask=patch_mask,
            patch_id=patch_id,
            event_id=event_id,
            label_type=label_type,
            y_start=y_start,
            x_start=x_start,
            qc_stats=qc_stats,
            domain="unknown",
            time_offset_hours=ms_data.time_offset_hours,
            satellite="unknown",
            built_up_fraction="unknown"
        )
        
        return {
            "patch_id": patch_id,
            "event_id": event_id,
            "path": patch_path,
            "label_type": label_type,
            "components": comps_present
        }, "passed"

    # Positives
    for comp in accepted:
        # Find centroid or bounds of component
        comp_mask = (mask_info['labeled_array'] == comp['id'])
        coords = np.argwhere(comp_mask)
        y_min, y_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        x_min, x_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        
        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2
        
        # Jitter so cy, cx is randomly placed within the patch (but must fit inside master)
        # patch covers [y_start, y_start+patch_px]
        # so y_start <= cy <= y_start+patch_px -> cy - patch_px <= y_start <= cy
        # And 0 <= y_start <= master_px - patch_px
        
        y_start_min = max(0, cy - patch_px + 10) # +10 to ensure some margin
        y_start_max = min(master_px - patch_px, cy - 10)
        if y_start_min > y_start_max:
            y_start_min, y_start_max = max(0, cy - patch_px), min(master_px - patch_px, cy)
            
        x_start_min = max(0, cx - patch_px + 10)
        x_start_max = min(master_px - patch_px, cx - 10)
        if x_start_min > x_start_max:
            x_start_min, x_start_max = max(0, cx - patch_px), min(master_px - patch_px, cx)
            
        # MULTIPLIER: Extract 3 patches per fire component
        for _ in range(3):
            y_start = random.randint(y_start_min, y_start_max)
            x_start = random.randint(x_start_min, x_start_max)
            
            res, reason = extract_patch(y_start, x_start, "positive")
            if res:
                patches_extracted.append(res)
            
    # Hard negative extraction from the same scene (if it doesn't contain fire)
    # Just try a random crop that doesn't intersect accepted/review
    for _ in range(6): # Try 6 hard negatives per scene
        y_start = random.randint(0, master_px - patch_px)
        x_start = random.randint(0, master_px - patch_px)
        
        # Check if it has fire or review required
        sub_mask = mask[y_start:y_start+patch_px, x_start:x_start+patch_px]
        if not np.any((sub_mask == 1) | (sub_mask == 2) | (sub_mask == 255)):
            res, reason = extract_patch(y_start, x_start, "hard_negative")
            if res:
                patches_extracted.append(res)
                
    if not accepted and not mask_info['review_required']:
        # This is a pure negative event
        y_start = master_px // 2 - patch_px // 2
        x_start = master_px // 2 - patch_px // 2
        res, reason = extract_patch(y_start, x_start, "negative")
        if res:
            patches_extracted.append(res)

    return patches_extracted
