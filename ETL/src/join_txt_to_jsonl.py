import os
import json
from tqdm import tqdm
import logging

# Log file in extraction directory
log_path = os.path.join('extraction', 'join_txt_to_jsonl.log')
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

jsonl_path = os.path.join('court_cases_output.jsonl')
txt_dir = os.path.join('results')
output_path = os.path.join('court_cases_output_with_text.jsonl')

logging.info('Starting TXT to JSONL join process.')

# Count total lines for progress bar if file is not too large
try:
    file_size = os.path.getsize(jsonl_path)
    if file_size < 100 * 1024 * 1024:  # 100MB threshold
        with open(jsonl_path, encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
    else:
        total_lines = None
        logging.info('JSONL file too large to count lines for tqdm total.')
except Exception as e:
    logging.error(f"Failed to count lines in JSONL file: {e}")
    total_lines = None

try:
    with open(jsonl_path, encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, total=total_lines, desc='Processing'):
            try:
                obj = json.loads(line)
            except Exception as e:
                logging.error(f"Failed to parse JSON: {e} | Line: {line[:100]}")
                continue
            pdf_filename = obj.get('pdf_filename')
            txt_filename = pdf_filename.replace('.pdf', '.txt') if pdf_filename else None
            txt_path = os.path.join(txt_dir, txt_filename) if txt_filename else None
            text = None
            if txt_path and os.path.exists(txt_path):
                try:
                    with open(txt_path, encoding='utf-8') as tf:
                        text = tf.read()
                except Exception as e:
                    logging.error(f"Failed to read TXT file {txt_filename}: {e}")
            else:
                if pdf_filename:
                    logging.warning(f"TXT file for {pdf_filename} not found.")
            obj['text'] = text
            try:
                fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
            except Exception as e:
                logging.error(f"Failed to write JSON object: {e} | Object: {str(obj)[:100]}")
except Exception as e:
    logging.critical(f"Failed to process JSONL file: {e}")
    raise

logging.info(f"Done. Output written to {output_path}")
print(f"Done. Output written to {output_path}") 