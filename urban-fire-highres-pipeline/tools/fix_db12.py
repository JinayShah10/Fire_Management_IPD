import glob
import os
import numpy as np
from tqdm import tqdm

def fix_dir(d):
    files = glob.glob(os.path.join(d, "*.npz"))
    for f in tqdm(files, desc=f"Fixing {d}"):
        data = dict(np.load(f, allow_pickle=True))
        stack = data.get('stack')
        if stack is None:
            # Maybe it's a master file with a different structure
            # Wait, master files might not have 'stack', they might have 'arrays' or just saved differently?
            pass
            
        if stack is not None:
            # stack[0] is B12, stack[7] is dB12
            b12 = stack[0]
            with np.errstate(divide='ignore', invalid='ignore'):
                db12 = 10 * np.log10(b12 + 1e-7)
            stack[7] = db12
            data['stack'] = stack
            np.savez_compressed(f, **data)

fix_dir('stage_030_dataset_builder/data/patches')
fix_dir('stage_030_dataset_builder/data/masters')
print("Done!")
