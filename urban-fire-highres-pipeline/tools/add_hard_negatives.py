import pandas as pd
import os
from datetime import datetime

HARD_NEGATIVES = [
    # (lat, lon, description)
    (35.5568, -115.4705, "Ivanpah Solar Power Facility, Nevada"),
    (36.8000, -2.7000, "Almeria Greenhouses, Spain"),
    (25.2100, 55.2700, "Dubai Industrial Areas"),
    (35.3000, -119.5000, "California Solar Farms"),
    (40.7128, -74.0060, "NYC High-Rise Reflective Roofs")
]

def main():
    csv_path = "stage_020_prescreening/data/events_prescreened.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    new_rows = []
    # Just use a recent date for all of them so we can fetch imagery
    dummy_date = "2023-06-01"
    dummy_time = "12:00"
    
    for lat, lon, desc in HARD_NEGATIVES:
        new_row = {
            "latitude": lat,
            "longitude": lon,
            "acq_date": dummy_date,
            "earliest_time": dummy_time,
            "firms_source": "MANUAL_HARD_NEGATIVE",
            "firms_request_id": "hard_neg",
            "firms_fetch_utc": "2023-06-01 12:00:00",
            "event_id": f"hard_neg_{lat}_{lon}".replace(".", "_").replace("-", "n"),
            "source_type": "HARD_NEGATIVE",
            "description": desc
        }
        new_rows.append(new_row)
        
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df, df_new], ignore_index=True)
    df_combined.to_csv(csv_path, index=False)
    print(f"Successfully added {len(HARD_NEGATIVES)} hard negatives to the dataset pipeline.")

if __name__ == "__main__":
    main()
