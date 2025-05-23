import json
from collections import defaultdict

# Path to the JSONL file
jsonl_path = 'scrapy/court_cases_output.jsonl'

# Dictionary to store 'nomor' and their line numbers
nomor_lines = defaultdict(list)

with open(jsonl_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f, 1):  # Line numbers start at 1
        try:
            data = json.loads(line)
            nomor = data.get('nomor')
            if nomor:
                nomor_lines[nomor].append(idx)
        except Exception as e:
            print(f"Error parsing line {idx}: {e}")

# Print duplicates
found = False
for nomor, lines in nomor_lines.items():
    if len(lines) > 1:
        found = True
        print(f"Duplicate nomor: {nomor} (lines: {lines})")

if not found:
    print("No duplicate 'nomor' found.") 