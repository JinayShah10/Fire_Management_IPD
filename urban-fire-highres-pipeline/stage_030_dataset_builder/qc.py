import numpy as np


def compute_qc_stats(ms_data, config):
    scl = ms_data.arrays['SCL']
    # Nodata fraction (where B12 is NaN, since NaN means nodata)
    b12 = ms_data.arrays['B12']
    
    total_pixels = b12.size
    nodata_mask = np.isnan(b12)
    nodata_frac = np.sum(nodata_mask) / total_pixels
    
    # Cloud mask based on SCL
    cloud_classes = config.qc.cloud_scl_classes
    cloud_mask = np.isin(scl, cloud_classes)
    cloud_frac = np.sum(cloud_mask) / total_pixels
    
    # Saturated fraction (using reflectance > 1.0 as a proxy for saturated or very bright)
    sat_mask = b12 > 1.0
    sat_frac = np.sum(sat_mask) / total_pixels
    
    return {
        "nodata_fraction": float(nodata_frac),
        "cloud_fraction": float(cloud_frac),
        "saturated_fraction": float(sat_frac)
    }

def passes_qc(qc_stats, config):
    if qc_stats["cloud_fraction"] > config.qc.max_cloud_frac:
        return False, "cloud_fraction_exceeded"
    if qc_stats["nodata_fraction"] > config.qc.max_nodata_frac:
        return False, "nodata_fraction_exceeded"
    return True, "passed"
