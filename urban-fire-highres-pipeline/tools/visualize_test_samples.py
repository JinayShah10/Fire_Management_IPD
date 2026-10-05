import glob
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
import json

def main():
    device = torch.device("cpu")
    model_path = "artifacts/unet_fire_best.pth"
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=8, classes=1)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open("artifacts/norm_stats.json", "r") as f:
        stats = json.load(f)
        mean = torch.tensor(stats['mean']).view(-1, 1, 1).float()
        std = torch.tensor(stats['std']).view(-1, 1, 1).float()

    patches_dir = "stage_030_dataset_builder/data/patches"
    npz_files = glob.glob(os.path.join(patches_dir, "*.npz"))
    
    test_files = [f for f in npz_files if "positive" in os.path.basename(f)]
    
    # Try to pick 5 unique events
    event_dict = {}
    for f in test_files:
        try:
            eid = f.split("/")[-1].split("_positive")[0]
            if eid not in event_dict:
                event_dict[eid] = []
            event_dict[eid].append(f)
        except:
            pass

    selected_events = random.sample(list(event_dict.keys()), min(5, len(event_dict)))
    
    out_dir = "artifacts/samples"
    os.makedirs(out_dir, exist_ok=True)

    for idx, eid in enumerate(selected_events):
        f = event_dict[eid][0] # Pick the first patch for this event
        
        data = np.load(f, allow_pickle=True)
        stack = data['stack'].astype(np.float32)
        mask = data['mask'].astype(np.float32)
        
        valid_px = (mask != 255)
        tgt = np.zeros_like(mask, dtype=np.uint8)
        tgt[(mask == 1) | (mask == 2)] = 1
        tgt[~valid_px] = 0

        input_tensor = torch.tensor(stack)
        input_tensor = (input_tensor - mean) / (std + 1e-8)
        input_tensor = torch.nan_to_num(input_tensor, nan=0.0)
        
        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0))
            pred_prob = torch.sigmoid(output).squeeze().numpy()

        swir = stack[0] # B12
        swir_norm = (swir - np.nanmin(swir)) / (np.nanmax(swir) - np.nanmin(swir) + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(swir_norm, cmap='hot')
        axes[0].set_title(f"Input SWIR (B12)")
        axes[0].axis('off')

        axes[1].imshow(tgt, cmap='gray')
        axes[1].set_title(f"Target Mask")
        axes[1].axis('off')

        axes[2].imshow(pred_prob, cmap='inferno')
        axes[2].set_title(f"Model Prediction")
        axes[2].axis('off')

        plt.suptitle(f"Event: {eid}")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"sample_{idx+1}_{eid}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
