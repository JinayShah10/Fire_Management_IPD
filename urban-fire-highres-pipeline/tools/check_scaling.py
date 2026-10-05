import glob
import os

import numpy as np

from shared.config import get_config


def check_scaling(dataset_dir="stage_030_dataset_builder/data/patches"):
    config = get_config()
    patch_files = glob.glob(os.path.join(dataset_dir, "*.npz"))
    
    if not patch_files:
        print("No patches found to check scaling.")
        return
        
    print(f"Checking scaling for {len(patch_files)} patches...")
    
    channels = ['B12', 'B11', 'B8A', 'B8', 'B4', 'B3', 'B2', 'dB12']
    
    # Accumulate samples for percentiles (random subset to fit in memory)
    # We will sample 10% of pixels from each patch to avoid memory issues
    
    fire_pixels = {ch: [] for ch in channels}
    nonfire_pixels = {ch: [] for ch in channels}
    
    for pf in patch_files:
        data = np.load(pf)
        stack = data['stack']
        mask = data['mask']
        
        # Fire vs Non-fire (1 = core, 2 = margin)
        is_fire = (mask == 1) | (mask == 2)
        is_nonfire = (mask == 0)
        
        # Sample pixels
        # Flatten and subsample 1 in 10
        for i, ch in enumerate(channels):
            ch_data = stack[i]
            
            # Fire
            fire_vals = ch_data[is_fire]
            if len(fire_vals) > 0:
                fire_pixels[ch].append(np.random.choice(fire_vals, size=max(1, len(fire_vals)//10), replace=False))
                
            # Nonfire
            nonfire_vals = ch_data[is_nonfire]
            if len(nonfire_vals) > 0:
                nonfire_pixels[ch].append(np.random.choice(nonfire_vals, size=max(1, len(nonfire_vals)//100), replace=False))
                
    caps = {
        'B12': config.normalization.reflectance_clip[1],
        'B11': config.normalization.reflectance_clip[1],
        'B8A': config.normalization.reflectance_clip[1],
        'B8': config.normalization.reflectance_clip[1],
        'B4': config.normalization.reflectance_clip[1],
        'B3': config.normalization.reflectance_clip[1],
        'B2': config.normalization.reflectance_clip[1],
        'dB12': config.normalization.db12_clip[1] if hasattr(config.normalization, 'db12_clip') else 2.0
    }
    
    for ch in channels:
        print(f"\n--- Channel {ch} ---")
        for name, pix_list in [("Fire", fire_pixels[ch]), ("Non-fire", nonfire_pixels[ch])]:
            if not pix_list:
                print(f"  {name}: No data")
                continue
            
            all_pix = np.concatenate(pix_list)
            all_pix = all_pix[~np.isnan(all_pix)] # drop nans
            
            if len(all_pix) == 0:
                print(f"  {name}: All NaN")
                continue
                
            p50 = np.percentile(all_pix, 50)
            p95 = np.percentile(all_pix, 95)
            p99 = np.percentile(all_pix, 99)
            p999 = np.percentile(all_pix, 99.9)
            pmax = np.max(all_pix)
            
            cap = caps[ch]
            sat_frac = np.mean(all_pix >= cap)
            
            print(f"  {name}: P50={p50:.4f}, P95={p95:.4f}, P99={p99:.4f}, P99.9={p999:.4f}, Max={pmax:.4f} | Saturation Frac: {sat_frac:.2%}")

if __name__ == "__main__":
    check_scaling()
