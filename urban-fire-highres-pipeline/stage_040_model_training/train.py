import argparse
import glob
import json
import os
import random

import numpy as np
import segmentation_models_pytorch as smp
import torch
from scipy import ndimage
from scipy.spatial.distance import cdist
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from shared.config import get_config
from stage_030_dataset_builder.split import get_deterministic_split


class FireDataset(Dataset):
    def __init__(self, npz_paths, mean, std, augment=False, config=None):
        self.paths = npz_paths
        self.mean = torch.tensor(mean).view(-1, 1, 1).float()
        self.std = torch.tensor(std).view(-1, 1, 1).float()
        self.augment = augment
        self.config = config
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        stack = data['stack'].astype(np.float32)
        mask = data['mask'].astype(np.float32)
        
        # 1 and 2 are fire. 0 is bg. 255 is ignore.
        target = np.zeros_like(mask)
        target[(mask == 1) | (mask == 2)] = 1.0
        target[mask == 255] = 255.0
        
        # Augmentations
        if self.augment:
            if random.random() > 0.5:
                stack = np.flip(stack, axis=2)
                target = np.flip(target, axis=1)
            if random.random() > 0.5:
                stack = np.flip(stack, axis=1)
                target = np.flip(target, axis=0)
            
            k = random.randint(0, 3)
            if k > 0:
                stack = np.rot90(stack, k=k, axes=(1, 2))
                target = np.rot90(target, k=k, axes=(0, 1))
                
            # dB12 dropout
            if random.random() < getattr(self.config.dataset, 'dB12_dropout', 0.20):
                # dB12 is at index 7 (CHANNEL_ORDER)
                stack[7, :, :] = 0.0
                
            # Spectral Jitter
            if random.random() < 0.5:
                jitter = np.random.uniform(0.9, 1.1, size=(stack.shape[0], 1, 1)).astype(np.float32)
                stack = stack * jitter

        stack = np.ascontiguousarray(stack)
        target = np.ascontiguousarray(target)
        
        stack_t = (torch.tensor(stack) - self.mean) / (self.std + 1e-8)
        stack_t = torch.nan_to_num(stack_t, nan=0.0)
        target_t = torch.tensor(target).unsqueeze(0)
        
        return stack_t, target_t, self.paths[idx]


def compute_component_f1(pred_prob, target_mask, threshold, match_dist_px):
    pred = (pred_prob > threshold).astype(np.uint8)
    tgt = (target_mask == 1).astype(np.uint8) # 1 is fire
    
    labeled_pred, num_pred = ndimage.label(pred)
    labeled_tgt, num_tgt = ndimage.label(tgt)
    
    if num_tgt == 0 and num_pred == 0:
        return 0, 0, 0 # TP, FP, FN
        
    if num_tgt == 0:
        return 0, num_pred, 0
    if num_pred == 0:
        return 0, 0, num_tgt
        
    cent_pred = ndimage.center_of_mass(pred, labeled_pred, range(1, num_pred+1))
    cent_tgt = ndimage.center_of_mass(tgt, labeled_tgt, range(1, num_tgt+1))
    
    if len(cent_pred) == 0 or len(cent_tgt) == 0:
        return 0, num_pred, num_tgt
        
    dists = cdist(cent_tgt, cent_pred)
    
    # Greedy match
    tp = 0
    matched_pred = set()
    for i in range(len(cent_tgt)):
        valid_preds = np.where(dists[i] <= match_dist_px)[0]
        for p in valid_preds:
            if p not in matched_pred:
                matched_pred.add(p)
                tp += 1
                break
                
    fp = num_pred - tp
    fn = num_tgt - tp
    return tp, fp, fn


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha, gamma, dice_w, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.dice = smp.losses.DiceLoss(mode='binary', ignore_index=ignore_index)
        self.dice_w = dice_w
        
    def forward(self, y_pred, y_true):
        # Native PyTorch Focal Loss (MPS safe)
        mask = y_true != self.ignore_index
        if mask.sum() > 0:
            yp = y_pred[mask]
            yt = y_true[mask]
            bce = nn.functional.binary_cross_entropy_with_logits(yp, yt, reduction='none')
            pt = torch.exp(-bce)
            fl = (self.alpha * (1 - pt) ** self.gamma * bce).mean()
        else:
            fl = 0.0
            
        dl = self.dice(y_pred, y_true)
        return (1.0 - self.dice_w) * fl + self.dice_w * dl


class FireBatchSampler(Sampler):
    def __init__(self, pos_indices, neg_indices, batch_size, pos_ratio=0.40):
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        
        self.num_pos_per_batch = max(1, int(self.batch_size * self.pos_ratio))
        self.num_neg_per_batch = self.batch_size - self.num_pos_per_batch
        
        # Determine number of batches by how many negative batches we can form (or positive)
        # We'll just loop based on total length / batch_size
        self.num_batches = (len(self.pos_indices) + len(self.neg_indices)) // self.batch_size
        if self.num_batches == 0:
            self.num_batches = 1
            
    def __iter__(self):
        pos_copy = list(self.pos_indices)
        neg_copy = list(self.neg_indices)
        random.shuffle(pos_copy)
        random.shuffle(neg_copy)
        
        for _ in range(self.num_batches):
            batch = []
            for _ in range(self.num_pos_per_batch):
                if not pos_copy: pos_copy = list(self.pos_indices); random.shuffle(pos_copy)
                if pos_copy: batch.append(pos_copy.pop())
            for _ in range(self.num_neg_per_batch):
                if not neg_copy: neg_copy = list(self.neg_indices); random.shuffle(neg_copy)
                if neg_copy: batch.append(neg_copy.pop())
            random.shuffle(batch)
            yield batch
            
    def __len__(self):
        return self.num_batches

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    config = get_config()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    patches_dir = "stage_030_dataset_builder/data/patches"
    masters_dir = "stage_030_dataset_builder/data/masters"
    npz_files = glob.glob(os.path.join(patches_dir, "*.npz"))
    
    if len(npz_files) == 0:
        print("No patches found.")
        return
        
    # Map event to scene
    event_to_scene = {}
    for mf in glob.glob(os.path.join(masters_dir, "*.npz")):
        try:
            d = np.load(mf, allow_pickle=True)
            eid = os.path.basename(mf).replace("_master.npz", "")
            event_to_scene[eid] = str(d['scene_id'])
        except:
            pass

    train_files, val_files, test_files = [], [], []
    pos_train, neg_train, hneg_train = [], [], []
    
    fire_pixels = 0
    total_pixels = 0
    
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        eid = str(d['event_id'])
        ltype = str(d['label_type'])
        
        scene_id = event_to_scene.get(eid, eid)
        split = get_deterministic_split(scene_id, config)
        
        if split == 'train':
            train_files.append(f)
            if ltype == 'positive': pos_train.append(f)
            elif ltype == 'hard_negative': hneg_train.append(f)
            else: neg_train.append(f)
            
            # Compute fire pixels for weighting
            m = d['mask']
            fire_pixels += np.sum((m == 1) | (m == 2))
            total_pixels += m.size
            
        elif split == 'val':
            val_files.append(f)
        else:
            test_files.append(f)
            
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    os.makedirs("artifacts", exist_ok=True)
    norm_stats_path = "artifacts/norm_stats.json"
    
    if not os.path.exists(norm_stats_path):
        print("Computing norm stats on TRAIN...")
        sum_ch = np.zeros(8)
        sq_sum_ch = np.zeros(8)
        cnt = np.zeros(8)
        for f in train_files:
            st = np.load(f)['stack']
            
            for c in range(8):
                valid = ~np.isnan(st[c])
                sum_ch[c] += np.sum(st[c][valid])
                sq_sum_ch[c] += np.sum(st[c][valid]**2)
                cnt[c] += np.sum(valid)
            
        mean = sum_ch / np.maximum(cnt, 1)
        std = np.sqrt(np.maximum(0, sq_sum_ch / np.maximum(cnt, 1) - mean**2))
        with open(norm_stats_path, "w") as jf:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, jf)
            
    with open(norm_stats_path, "r") as jf:
        stats = json.load(jf)
        mean = stats['mean']
        std = stats['std']
        
    print(f"Norm Mean: {mean}\nNorm Std: {std}")
    
    # Class weighting
    pos_ratio = getattr(config.train, 'sampler_pos_ratio', 0.40)
    f_frac = max(1e-6, fire_pixels / max(1, total_pixels))
    # sampler boosts fire
    f_frac_effective = max(f_frac, pos_ratio)
    alpha = min(50.0, max(1.0, np.sqrt(1.0 / f_frac_effective)))
    print(f"Fire fraction (raw): {f_frac:.4f}, Effective: {f_frac_effective:.4f}, alpha weight: {alpha:.2f}")
    
    # Custom sampler
    pos_idxs = [i for i, f in enumerate(train_files) if f in pos_train]
    neg_idxs = [i for i, f in enumerate(train_files) if f not in pos_train]
    if not pos_idxs: pos_idxs = [0]
    if not neg_idxs: neg_idxs = [0]
    
    batch_sampler = FireBatchSampler(pos_idxs, neg_idxs, config.train.batch_size, pos_ratio=pos_ratio)
    
    train_dataset = FireDataset(train_files, mean, std, augment=True, config=config)
    val_dataset = FireDataset(val_files, mean, std, augment=False, config=config)
    
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)
    
    model = smp.Unet(
        encoder_name=config.model.encoder,
        encoder_weights="imagenet" if config.model.pretrained else None,
        in_channels=config.model.in_channels,
        classes=1
    )
    model_save_path = "artifacts/unet_fire_best.pth"
    if os.path.exists(model_save_path):
        print(f"Resuming weights from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    model = model.to(device)
    
    criterion = FocalDiceLoss(
        alpha=alpha, 
        gamma=config.loss.focal_gamma, 
        dice_w=config.loss.dice_weight, 
        ignore_index=config.labels.ignore_index
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.train.lr), weight_decay=float(config.train.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    best_f1 = -1.0
    try:
        if os.path.exists("artifacts/threshold.json"):
            with open("artifacts/threshold.json", "r") as f:
                best_f1 = json.load(f)["val_f1"]
                print(f"Resuming with previous best F1: {best_f1:.4f}")
    except Exception as e:
        print(f"Could not load previous best F1: {e}")
    model_save_path = "artifacts/unet_fire_best.pth"
    
    epochs = config.train.epochs
    warmup_epochs = getattr(config.train, 'warmup_epochs', 2)
    patience = config.train.early_stop_patience
    no_improve = 0
    
    thresholds = getattr(config.train, 'validation_thresholds', [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995])
    
    for epoch in range(epochs):
        # Warmup
        if epoch < warmup_epochs:
            for param in model.encoder.parameters():
                param.requires_grad = False
        else:
            for param in model.encoder.parameters():
                param.requires_grad = True
                
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}")
        for X, y, _ in pbar:
            X, y = X.to(device), y.to(device)
            
            # MixUp Augmentation
            if epoch < epochs - 5 and random.random() < 0.3: # Disable mixup in last 5 epochs
                lam = np.random.beta(0.2, 0.2)
                index = torch.randperm(X.size(0)).to(device)
                X = lam * X + (1 - lam) * X[index, :]
                y = lam * y + (1 - lam) * y[index, :]
                
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        all_tp = {t: 0 for t in thresholds}
        all_fp = {t: 0 for t in thresholds}
        all_fn = {t: 0 for t in thresholds}
        
        with torch.no_grad():
            for X, y, _ in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item() * X.size(0)
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                yt_np = y.cpu().numpy()
                
                for b in range(X.size(0)):
                    p_prob = probs[b, 0]
                    tgt = yt_np[b, 0]
                    
                    for t in thresholds:
                        tp, fp, fn = compute_component_f1(p_prob, tgt, t, config.eval.match_dist_px)
                        all_tp[t] += tp
                        all_fp[t] += fp
                        all_fn[t] += fn
                        
        val_loss /= len(val_dataset)
        
        # Find best threshold this epoch
        best_t_epoch = 0.5
        best_f1_epoch = -1.0
        best_prec_epoch = 0.0
        best_rec_epoch = 0.0
        for t in thresholds:
            tp, fp, fn = all_tp[t], all_fp[t], all_fn[t]
            prec = tp / (tp + fp) if tp + fp > 0 else 0
            rec = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
            if f1 > best_f1_epoch:
                best_f1_epoch = f1
                best_t_epoch = t
                best_prec_epoch = prec
                best_rec_epoch = rec
                
        print(f"Ep {epoch+1} - TL: {train_loss:.4f} | VL: {val_loss:.4f} | Best F1: {best_f1_epoch:.4f} (Prec: {best_prec_epoch:.4f}, Rec: {best_rec_epoch:.4f}) @ T={best_t_epoch}")
        
        if best_f1_epoch > best_f1:
            best_f1 = best_f1_epoch
            torch.save(model.state_dict(), model_save_path)
            with open("artifacts/threshold.json", "w") as f:
                json.dump({"best_epoch": epoch+1, "threshold": best_t_epoch, "val_f1": best_f1}, f)
            print("  -> Saved new best model!")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break
                
        # Step the scheduler based on best validation F1
        scheduler.step(best_f1_epoch)

if __name__ == "__main__":
    main()
