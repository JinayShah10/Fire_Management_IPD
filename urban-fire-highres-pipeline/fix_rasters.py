import os
import subprocess
import glob

cache_dir = "stage_015_urban_filtering/cache/rasters"
files = glob.glob(f"{cache_dir}/*.tif")
print(f"Validating {len(files)} TIFF files for corruption...")

bad_files = 0
for i, f in enumerate(files):
    if i % 100 == 0:
        print(f"Checked {i}/{len(files)}...")
    
    # Run in subprocess to isolate segfaults!
    code = f"""
import rasterio
try:
    with rasterio.open('{f}') as src:
        # Read a tiny piece to force data access
        _ = src.read(1, window=rasterio.windows.Window(0, 0, 1, 1))
except Exception as e:
    exit(1)
"""
    result = subprocess.run(["python3", "-c", code], capture_output=True)
    if result.returncode != 0:
        print(f"CORRUPT FILE FOUND: {f} (Return code: {result.returncode})")
        os.remove(f)
        bad_files += 1

print(f"Done! Deleted {bad_files} corrupt files.")
