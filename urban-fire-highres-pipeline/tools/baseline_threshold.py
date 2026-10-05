import glob
import os

import numpy as np
from scipy import ndimage

from shared.config import get_config
from stage_030_dataset_builder.split import get_deterministic_split
from stage_050_evaluation.evaluate import compute_component_metrics


def apply_swir_rule(stack, ignore_mask):
    # stack: ['B12', 'B11', 'B8A', 'B8', 'B4', 'B3', 'B2', 'dB12']
    b12 = stack[0]
    b8a = stack[2]
    
    ratio = b12 / (b8a + 1e-6)
    
    seeds = (b12 >= 0.35) & (ratio >= 4.0)
    growth_mask = (b12 >= 0.25) & (ratio >= 2.0)
    
    struct = ndimage.generate_binary_structure(2, 1) # 4-connectivity
    step1 = ndimage.binary_dilation(seeds, structure=struct)
    step2 = ndimage.binary_dilation(step1, structure=struct)
    
    final_pred = step2 & growth_mask
    final_pred[ignore_mask] = 0
    return final_pred.astype(np.uint8)

def main():
    config = get_config()
    
    patches_dir = "stage_030_dataset_builder/data/patches"
    masters_dir = "stage_030_dataset_builder/data/masters"
    npz_files = glob.glob(os.path.join(patches_dir, "*.npz"))
    
    event_to_scene = {}
    for mf in glob.glob(os.path.join(masters_dir, "*.npz")):
        try:
            d = np.load(mf, allow_pickle=True)
            eid = os.path.basename(mf).replace("_master.npz", "")
            event_to_scene[eid] = str(d['scene_id'])
        except:
            pass

    val_files = []
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        eid = str(d['event_id'])
        scene_id = event_to_scene.get(eid, eid)
        # Prompt says "tuned on VALIDATION only", so evaluate on validation
        if get_deterministic_split(scene_id, config) == 'val':
            val_files.append(f)
            
    print(f"Evaluating RULE-BASED BASELINE on {len(val_files)} VALIDATION images...")
    
    total_tp, total_fp, total_fn = 0, 0, 0
    total_area_m2 = 0
    
    for f in val_files:
        data = np.load(f)
        stack = data['stack'].astype(np.float32)
        mask = data['mask'].astype(np.float32)
        
        valid_px = (mask != config.labels.ignore_index)
        ignore_mask = ~valid_px
        
        pred = apply_swir_rule(stack, ignore_mask)
        
        tgt = np.zeros_like(mask, dtype=np.uint8)
        tgt[(mask == 1) | (mask == 2)] = 1
        tgt[ignore_mask] = 0
        
        tp, fp, fn = compute_component_metrics(pred, tgt, config.eval.match_dist_px)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        total_area_m2 += (stack.shape[1] * config.grid.resolution_m) * (stack.shape[2] * config.grid.resolution_m)

    prec = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    rec = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    
    total_area_km2 = total_area_m2 / 1_000_000.0
    fa_per_100km2 = (total_fp / total_area_km2) * 100 if total_area_km2 > 0 else 0
    
    print("\n--- WEAK-LABEL BASELINE (SWIR Seed/Grow Rule) ---")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision [WEAK-LABEL]: {prec:.4f}")
    print(f"Recall [WEAK-LABEL]:    {rec:.4f}")
    print(f"F1 Score [WEAK-LABEL]:  {f1:.4f}")
    print(f"\nTotal Area: {total_area_km2:.2f} km^2")
    print(f"False Alarms per 100 km^2 [WEAK-LABEL]: {fa_per_100km2:.4f}")
    print("\nWARNING: Baseline-vs-weak-label scores are CIRCULAR (labels come from the same kind of rule) and must not be reported as accuracy. The baseline is meaningful only on the gold set.")

if __name__ == "__main__":
    main()
