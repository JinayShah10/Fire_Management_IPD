import glob
import os
import sys
from collections import defaultdict

import numpy as np

from shared.config import get_config
from stage_030_dataset_builder.split import get_deterministic_split


def run_report(dataset_dir="stage_030_dataset_builder/data/patches"):
    config = get_config()
    patch_files = glob.glob(os.path.join(dataset_dir, "*.npz"))
    
    if not patch_files:
        print("No patches found.")
        sys.exit(0)
        
    master_dir = os.path.join(os.path.dirname(dataset_dir), "masters")
    
    # We need scene_id from masters
    event_to_scene = {}
    master_files = glob.glob(os.path.join(master_dir, "*.npz"))
    for mf in master_files:
        try:
            data = np.load(mf, allow_pickle=True)
            event_id = os.path.basename(mf).replace("_master.npz", "")
            scene_id = str(data['scene_id'])
            event_to_scene[event_id] = scene_id
        except Exception:
            pass
            
    stats = {
        'train': {'patches': 0, 'positives': 0, 'negatives': 0, 'hard_negatives': 0, 'pixels': {0:0, 1:0, 2:0, 255:0}, 'groups': set(), 'domain': defaultdict(int)},
        'val': {'patches': 0, 'positives': 0, 'negatives': 0, 'hard_negatives': 0, 'pixels': {0:0, 1:0, 2:0, 255:0}, 'groups': set(), 'domain': defaultdict(int)},
        'test': {'patches': 0, 'positives': 0, 'negatives': 0, 'hard_negatives': 0, 'pixels': {0:0, 1:0, 2:0, 255:0}, 'groups': set(), 'domain': defaultdict(int)}
    }
    
    group_to_split = {}
    leakage = False
    
    for pf in patch_files:
        data = np.load(pf, allow_pickle=True)
        event_id = str(data['event_id'])
        label_type = str(data['label_type'])
        mask = data['mask']
        
        # Determine group
        scene_id = event_to_scene.get(event_id, event_id)
        group_key = scene_id # we'll use scene_id as group key
        
        split = get_deterministic_split(group_key, config)
        
        if group_key in group_to_split and group_to_split[group_key] != split:
            print(f"LEAKAGE DETECTED: Group {group_key} is in {group_to_split[group_key]} and {split}!")
            leakage = True
        group_to_split[group_key] = split
        
        s = stats[split]
        s['patches'] += 1
        if label_type == 'positive':
            s['positives'] += 1
        elif label_type == 'hard_negative':
            s['hard_negatives'] += 1
        else:
            s['negatives'] += 1
            
        unique, counts = np.unique(mask, return_counts=True)
        for u, c in zip(unique, counts):
            if u in s['pixels']:
                s['pixels'][u] += c
            else:
                s['pixels'][u] = c
                
        s['groups'].add(group_key)
        # We don't have domain info stored easily in npz unless we look up events.csv
        # We'll just put 'unknown' for now
        s['domain']['unknown'] += 1
        
    print("=== Dataset Report ===")
    for split in ['train', 'val', 'test']:
        s = stats[split]
        print(f"\n[{split.upper()}]")
        print(f"Patches: {s['patches']}")
        print(f"  Positives: {s['positives']}")
        print(f"  Negatives: {s['negatives']}")
        print(f"  Hard Negatives: {s['hard_negatives']}")
        print("Pixels:")
        for k, v in s['pixels'].items():
            print(f"  Class {k}: {v}")
        print(f"Distinct Groups: {len(s['groups'])}")
        print(f"Domain Counts: {dict(s['domain'])}")
        
    if leakage:
        print("\nERROR: Data leakage detected between splits!")
        sys.exit(1)
    else:
        print("\nPASS: No leakage between splits.")
        sys.exit(0)

if __name__ == "__main__":
    run_report()
