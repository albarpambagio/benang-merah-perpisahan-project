import sys
import json
from datetime import datetime
from zoneinfo import ZoneInfo

# Usage: python convert_timestamp_to_local.py input.jsonl output.jsonl
if len(sys.argv) != 3:
    print("Usage: python convert_timestamp_to_local.py input.jsonl output.jsonl")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

jakarta = ZoneInfo('Asia/Jakarta')

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        if not line.strip():
            continue
        item = json.loads(line)
        ts = item.get('timestamp')
        if ts:
            try:
                # Parse as UTC and set tzinfo if missing
                dt_utc = datetime.fromisoformat(ts)
                if dt_utc.tzinfo is None:
                    dt_utc = dt_utc.replace(tzinfo=ZoneInfo('UTC'))
                # Convert to Asia/Jakarta
                dt_jakarta = dt_utc.astimezone(jakarta)
                item['timestamp_local'] = dt_jakarta.isoformat()
            except Exception as e:
                item['timestamp_local'] = f"ERROR: {e}"
        json.dump(item, outfile, ensure_ascii=False)
        outfile.write('\n') 