import argparse
import json
import os

import numpy as np
import segmentation_models_pytorch as smp
import torch

from shared.config import get_config


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

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    model = smp.Unet(
        encoder_name=config.model.encoder,
        encoder_weights=None,
        in_channels=config.model.in_channels,
        classes=1
    )
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    with open("artifacts/norm_stats.json", "r") as f:
        stats = json.load(f)
        mean = torch.tensor(stats['mean']).view(-1, 1, 1).float().to(device)
        std = torch.tensor(stats['std']).view(-1, 1, 1).float().to(device)

    gold_dir = "artifacts/gold"
    gold_json_path = os.path.join(gold_dir, "gold_labels.json")
    
    if not os.path.exists(gold_json_path):
        print(f"Error: {gold_json_path} not found. Run make_gold_set.py first.")
        return
        
    with open(gold_json_path, "r") as f:
        gold_dict = json.load(f)
        
    print(f"Evaluating on {len(gold_dict)} GOLD patches...")
    
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    
    with torch.no_grad():
        for patch_id, true_label in gold_dict.items():
            f = os.path.join(gold_dir, f"{patch_id}.npz")
            if not os.path.exists(f):
                continue
                
            data = np.load(f)
            stack = data['stack'].astype(np.float32)
            
            input_tensor = torch.tensor(stack).to(device)
            input_tensor = (input_tensor - mean) / (std + 1e-8)
            input_tensor = input_tensor.unsqueeze(0)
            
            output = model(input_tensor)
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
            
            pred = (pred_prob > args.threshold).astype(np.uint8)
            
            pred_label = 1 if np.any(pred == 1) else 0
            
            if pred_label == 1 and true_label == 1: total_tp += 1
            elif pred_label == 1 and true_label == 0: total_fp += 1
            elif pred_label == 0 and true_label == 1: total_fn += 1
            else: total_tn += 1

    prec = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    rec = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    acc = (total_tp + total_tn) / len(gold_dict) if len(gold_dict) > 0 else 0
    
    print("\n--- GOLD SET EVALUATION [Patch-Level] ---")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}, TN: {total_tn}")
    print(f"Precision [GOLD]: {prec:.4f}")
    print(f"Recall [GOLD]:    {rec:.4f}")
    print(f"F1 Score [GOLD]:  {f1:.4f}")
    print(f"Accuracy [GOLD]:  {acc:.4f}")

if __name__ == "__main__":
    main()
