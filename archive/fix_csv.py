import csv
import os
import re

input_path = 'data/annotated_divorce_reason_sample.csv'
output_path = 'data/annotated_divorce_reason_sample_fixed.csv'
review_output_path = 'data/annotated_divorce_reason_sample_for_review.csv'

# Helper to check if value is 0 or 1
is_valid_label = lambda x: x in ('0', '1')

def clean_label(val):
    return val if is_valid_label(val) else ''

affected_rows = []

with open(input_path, 'r', encoding='utf-8') as infile, \
     open(output_path, 'w', encoding='utf-8', newline='') as outfile, \
     open(review_output_path, 'w', encoding='utf-8', newline='') as reviewfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    review_writer = csv.writer(reviewfile)
    header = next(reader)
    writer.writerow(['reason_text', 'label', 'manual_label'])
    review_writer.writerow(['reason_text', 'label'])
    for row in reader:
        # Remove empty strings at the end
        while row and row[-1] == '':
            row = row[:-1]
        if len(row) < 2:
            continue  # skip malformed
        if len(row) == 2:
            label = clean_label(row[1])
            manual_label = ''
            if not is_valid_label(row[1]):
                affected_rows.append(row)
            writer.writerow([row[0], label, manual_label])
            review_writer.writerow([row[0], label])
        elif len(row) == 3:
            label = clean_label(row[1])
            manual_label = clean_label(row[2])
            if not is_valid_label(row[1]) or (row[2] and not is_valid_label(row[2])):
                affected_rows.append(row)
            writer.writerow([row[0], label, manual_label])
            review_writer.writerow([row[0], label])
        else:
            # Merge all but last two as reason_text
            reason_text = ','.join(row[:-2])
            label = clean_label(row[-2])
            manual_label = clean_label(row[-1])
            if not is_valid_label(row[-2]) or (row[-1] and not is_valid_label(row[-1])):
                affected_rows.append(row)
            writer.writerow([reason_text, label, manual_label])
            review_writer.writerow([reason_text, label])

if affected_rows:
    print('Rows with non-numeric labels found and fixed:')
    for r in affected_rows:
        print(r)

# Replace the original file with the fixed one
os.replace(output_path, input_path)

input_path = 'data/annotated_divorce_reason_sample_for_review.csv'
labelstudio_output_path = 'data/annotated_divorce_reason_sample_for_labelstudio.csv'

with open(input_path, 'r', encoding='utf-8') as infile, \
     open(labelstudio_output_path, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
    header = next(reader)
    writer.writerow(['reason_text', 'label'])
    for row in reader:
        # Remove empty trailing fields
        while row and row[-1] == '':
            row = row[:-1]
        if len(row) < 2:
            continue
        # Merge all but last as reason_text, last as label
        reason_text = ','.join(row[:-1]).strip()
        label = row[-1].strip()
        writer.writerow([reason_text, label]) 