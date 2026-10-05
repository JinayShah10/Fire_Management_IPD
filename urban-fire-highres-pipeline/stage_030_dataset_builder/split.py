import hashlib
import logging

logger = logging.getLogger(__name__)

def get_deterministic_split(group_key, config):
    md5_hash = hashlib.md5(group_key.encode('utf-8')).hexdigest()
    hash_int = int(md5_hash[:8], 16)
    
    val_thresh = int(0xffffffff * config.dataset.split_ratios.train)
    test_thresh = val_thresh + int(0xffffffff * config.dataset.split_ratios.val)
    
    if hash_int < val_thresh:
        return 'train'
    elif hash_int < test_thresh:
        return 'val'
    else:
        return 'test'

def get_group_key(ms_data, event, config):
    if ms_data and ms_data.scene_id:
        return ms_data.scene_id
    else:
        # Fallback to spatial-temporal bucket
        lat = event['latitude']
        lon = event['longitude']
        dt = event['datetime']
        deg = config.dataset.group_cell_deg
        
        lat_bucket = int(lat / deg) * deg
        lon_bucket = int(lon / deg) * deg
        date_str = dt.strftime("%Y-%m-%d") if dt else "nodate"
        
        return f"bucket_{lat_bucket:.1f}_{lon_bucket:.1f}_{date_str}"
