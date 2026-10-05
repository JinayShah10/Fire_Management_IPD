import os
import pandas as pd
import numpy as np
import subprocess

df = pd.read_csv("stage_010_data_ingestion/data/events.csv")
df['grid_lon'] = np.floor(df['longitude'] / 0.1) * 0.1
df['grid_lat'] = np.floor(df['latitude'] / 0.1) * 0.1
groups = list(df.groupby(['grid_lat', 'grid_lon']))

print("Scanning groups around 19700-19900 for segfaults...")
for i in range(19750, 19850):
    if i >= len(groups): break
    
    code = f"""
import sys
sys.path.append('stage_015_urban_filtering')
from urban_filter import process_group
import pandas as pd
import numpy as np
import json
import argparse

df = pd.read_csv("stage_010_data_ingestion/data/events.csv")
df['grid_lon'] = np.floor(df['longitude'] / 0.1) * 0.1
df['grid_lat'] = np.floor(df['latitude'] / 0.1) * 0.1
groups = list(df.groupby(['grid_lat', 'grid_lon']))
group_df = groups[{i}][1]

class Args:
    pass
args = Args()
args.min_ghsl_urban_ratio = 0.05
args.min_built_up_ratio = 0.05
args.worldcover_radius_km = 1.0
args.min_connected_area_m2 = 50000

# We don't have the exact lulc/wc paths here easily without STAC discovery...
# Wait, STAC cache has the bbox.
"""
