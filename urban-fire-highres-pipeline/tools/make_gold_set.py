import glob
import json
import os
import random
import shutil

import numpy as np

from shared.config import get_config
from stage_030_dataset_builder.split import get_deterministic_split


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

    test_files = []
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        eid = str(d['event_id'])
        scene_id = event_to_scene.get(eid, eid)
        if get_deterministic_split(scene_id, config) == 'test':
            test_files.append(f)
            
    random.seed(42)
    gold_size = getattr(config.eval, 'gold_set_size', 150)
    if len(test_files) < gold_size:
        print(f"Warning: only {len(test_files)} test patches available, using all.")
        sample_files = test_files
    else:
        sample_files = random.sample(test_files, gold_size)
        
    gold_dir = "artifacts/gold"
    os.makedirs(gold_dir, exist_ok=True)
    
    gold_dict = {}
    
    for i, f in enumerate(sample_files):
        patch_id = os.path.basename(f).replace(".npz", "")
        dest = os.path.join(gold_dir, f"{patch_id}.npz")
        shutil.copy2(f, dest)
        gold_dict[patch_id] = 0 # 0 = no fire, 1 = fire
        
    with open(os.path.join(gold_dir, "gold_labels.json"), "w") as f:
        json.dump(gold_dict, f, indent=2)
        
    print(f"Copied {len(sample_files)} patches to {gold_dir}.")
    print("Created gold_labels.json. Please update it with human labels (0=no fire, 1=fire).")

if __name__ == "__main__":
    main()
