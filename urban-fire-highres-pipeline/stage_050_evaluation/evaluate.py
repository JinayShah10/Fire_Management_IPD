import argparse
import glob
import json
import os

import numpy as np
import segmentation_models_pytorch as smp
import torch
from scipy import ndimage
from scipy.spatial.distance import cdist

from shared.config import get_config
from stage_030_dataset_builder.split import get_deterministic_split


def compute_component_metrics(pred, tgt, match_dist_px):
    labeled_pred, num_pred = ndimage.label(pred)
    labeled_tgt, num_tgt = ndimage.label(tgt)
    
    if num_tgt == 0 and num_pred == 0:
        return 0, 0, 0
    if num_tgt == 0:
        return 0, num_pred, 0
    if num_pred == 0:
        return 0, 0, num_tgt
        
    cent_pred = ndimage.center_of_mass(pred, labeled_pred, range(1, num_pred+1))
    cent_tgt = ndimage.center_of_mass(tgt, labeled_tgt, range(1, num_tgt+1))
    
    cent_pred = [c for c in cent_pred if not np.isnan(c[0])]
    cent_tgt = [c for c in cent_tgt if not np.isnan(c[0])]
    
    num_pred = len(cent_pred)
    num_tgt = len(cent_tgt)
    
    if num_pred == 0 or num_tgt == 0:
        return 0, num_pred, num_tgt
        
    dists = cdist(cent_tgt, cent_pred)
    tp = 0
    matched_pred = set()
    for i in range(num_tgt):
        valid = np.where(dists[i] <= match_dist_px)[0]
        for p in valid:
            if p not in matched_pred:
                matched_pred.add(p)
                tp += 1
                break
                
    fp = num_pred - tp
    fn = num_tgt - tp
    return tp, fp, fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="artifacts/unet_fire_best.pth")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    config = get_config()
    
    if args.threshold is None:
        try:
            with open("artifacts/threshold.json", "r") as f:
                args.threshold = json.load(f)["threshold"]
        except:
            args.threshold = 0.5
            print("No artifacts/threshold.json found, using 0.5")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    model = smp.Unet(
        encoder_name=config.model.encoder,
        encoder_weights=None,
        in_channels=config.model.in_channels,
        classes=1
    )
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    if not os.path.exists(args.model_path):
        print(f"Warning: model not found at {args.model_path}, using untrained weights.")
    model = model.to(device)
    model.eval()

    with open("artifacts/norm_stats.json", "r") as f:
        stats = json.load(f)
        mean = torch.tensor(stats['mean']).view(-1, 1, 1).float().to(device)
        std = torch.tensor(stats['std']).view(-1, 1, 1).float().to(device)

    patches_dir = "stage_030_dataset_builder/data/patches"
    masters_dir = "stage_030_dataset_builder/data/masters"
    npz_files = glob.glob(os.path.join(patches_dir, "*.npz"))
    
    event_to_scene = {}
    event_to_offset = {}
    for mf in glob.glob(os.path.join(masters_dir, "*.npz")):
        try:
            d = np.load(mf, allow_pickle=True)
            eid = os.path.basename(mf).replace("_master.npz", "")
            event_to_scene[eid] = str(d['scene_id'])
            event_to_offset[eid] = float(d['time_offset_hours']) if 'time_offset_hours' in d else 0.0
        except:
            pass

    test_files = []
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        eid = str(d['event_id'])
        scene_id = event_to_scene.get(eid, eid)
        if get_deterministic_split(scene_id, config) == 'test':
            test_files.append(f)
            
    print(f"Evaluating on {len(test_files)} TEST images using threshold {args.threshold}...")
    
    total_tp, total_fp, total_fn = 0, 0, 0
    total_area_m2 = 0
    
    slices = {
        'cloud': {'0%': [0,0,0], '<5%': [0,0,0], '>=5%': [0,0,0]},
        'time_offset': {'0-2h': [0,0,0], '2-12h': [0,0,0], '12h+': [0,0,0]},
        'domain': {'urban': [0,0,0], 'rural': [0,0,0], 'unknown': [0,0,0]}
    }
    
    with torch.no_grad():
        for f in test_files:
            data = np.load(f, allow_pickle=True)
            stack = data['stack'].astype(np.float32)
            mask = data['mask'].astype(np.float32)
            eid = str(data['event_id'])
            qc = data['qc_stats'].item() if 'qc_stats' in data else {'cloud_fraction': 0.0}
            
            c_frac = qc.get('cloud_fraction', 0.0)
            if c_frac == 0: c_bucket = '0%'
            elif c_frac < 0.05: c_bucket = '<5%'
            else: c_bucket = '>=5%'
            
            t_off = event_to_offset.get(eid, 0.0)
            if t_off <= 2: t_bucket = '0-2h'
            elif t_off <= 12: t_bucket = '2-12h'
            else: t_bucket = '12h+'
            
            d_bucket = 'unknown' # Domain is tricky without events.csv, assume unknown for now
            
            valid_px = (mask != config.labels.ignore_index)
            
            input_tensor = torch.tensor(stack).to(device)
            input_tensor = (input_tensor - mean) / (std + 1e-8)
            input_tensor = torch.nan_to_num(input_tensor, nan=0.0)
            input_tensor = input_tensor.unsqueeze(0)
            
            output = model(input_tensor)
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
            
            pred = (pred_prob > args.threshold).astype(np.uint8)
            pred[~valid_px] = 0 # Ignore predictions in 255 regions
            
            tgt = np.zeros_like(mask, dtype=np.uint8)
            tgt[(mask == 1) | (mask == 2)] = 1
            tgt[~valid_px] = 0
            
            tp, fp, fn = compute_component_metrics(pred, tgt, config.eval.match_dist_px)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            for b_name, b_val in [('cloud', c_bucket), ('time_offset', t_bucket), ('domain', d_bucket)]:
                slices[b_name][b_val][0] += tp
                slices[b_name][b_val][1] += fp
                slices[b_name][b_val][2] += fn
            
            total_area_m2 += (stack.shape[1] * config.grid.resolution_m) * (stack.shape[2] * config.grid.resolution_m)

    prec = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    rec = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    
    total_area_km2 = total_area_m2 / 1_000_000.0
    fa_per_100km2 = (total_fp / total_area_km2) * 100 if total_area_km2 > 0 else 0
    
    print("\n--- WEAK-LABEL COMPONENT METRICS ---")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nTotal Area: {total_area_km2:.2f} km^2")
    print(f"False Alarms per 100 km^2: {fa_per_100km2:.4f}")
    
    report_path = "artifacts/evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("# Model Evaluation Report\n\n")
        f.write("## Overall WEAK-LABEL Metrics\n")
        f.write(f"- Precision: {prec:.4f}\n")
        f.write(f"- Recall: {rec:.4f}\n")
        f.write(f"- F1 Score: {f1:.4f}\n")
        f.write(f"- False Alarms per 100 km^2: {fa_per_100km2:.4f}\n\n")
        
        f.write("## Slices Table\n")
        f.write("| Category | Bucket | TP | FP | FN | Precision | Recall | F1 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for cat, buckets in slices.items():
            for b_name, (tp, fp, fn) in buckets.items():
                p = tp / (tp + fp) if tp + fp > 0 else 0
                r = tp / (tp + fn) if tp + fn > 0 else 0
                f_score = 2 * p * r / (p + r) if p + r > 0 else 0
                f.write(f"| {cat} | {b_name} | {tp} | {fp} | {fn} | {p:.4f} | {r:.4f} | {f_score:.4f} |\n")
    print(f"\nSaved Markdown Report to {report_path}")

if __name__ == "__main__":
    main()
