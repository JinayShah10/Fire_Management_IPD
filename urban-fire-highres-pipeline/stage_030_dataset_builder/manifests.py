import csv
import json
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

def update_manifest(filepath, event_id, new_rows):
    """
    Idempotent update: Replaces all rows matching event_id with new_rows.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    is_csv = filepath.endswith('.csv')
    existing_rows = []
    
    if os.path.exists(filepath):
        if is_csv:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
        else:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        existing_rows.append(json.loads(line))
                        
    # Filter out the old rows for this event_id
    filtered_rows = [row for row in existing_rows if row.get('event_id') != event_id]
    
    # Append the new rows
    filtered_rows.extend(new_rows)
    
    # Write atomically
    fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(filepath))
    try:
        with os.fdopen(fd, 'w') as f:
            if is_csv:
                if filtered_rows:
                    keys = filtered_rows[0].keys()
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(filtered_rows)
            else:
                for row in filtered_rows:
                    f.write(json.dumps(row) + '\n')
                    
        os.replace(temp_path, filepath)
    except Exception as e:
        os.unlink(temp_path)
        raise e
