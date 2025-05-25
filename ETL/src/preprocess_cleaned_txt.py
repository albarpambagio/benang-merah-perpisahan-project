import os
import argparse
from utils.text_cleaning import remove_boilerplate, normalize_whitespace
from typing import List

# Patterns to remove (boilerplate)
BOILERPLATE_PATTERNS: List[str] = [
    r"Kepaniteraan Mahkamah Agung Republik Indonesia[\s\S]*?harap segera hubungi Kepaniteraan Mahkamah Agung RI melalui :",
    r"Untuk Salinantera Pengadilan Agama Ban.*$",
]

def preprocess_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = remove_boilerplate(text, BOILERPLATE_PATTERNS)
    text = normalize_whitespace(text)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

def process_one(args):
    in_path, out_path = args
    preprocess_file(in_path, out_path)
    return os.path.basename(in_path), os.path.basename(out_path)

def main():
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description='Preprocess all cleaned .txt files in a folder (remove boilerplate, normalize whitespace).')
    parser.add_argument('--input-dir', required=True, help='Directory containing cleaned .txt files')
    parser.add_argument('--output-dir', required=True, help='Directory to save preprocessed .txt files')
    parser.add_argument('--jobs', type=int, default=4, help='Number of parallel workers (default: 4)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = []
    for fname in os.listdir(args.input_dir):
        if fname.endswith('_cleaned.txt'):
            in_path = os.path.join(args.input_dir, fname)
            out_name = fname.replace('_cleaned.txt', '.txt')
            out_path = os.path.join(args.output_dir, out_name)
            tasks.append((in_path, out_path))

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = list(tqdm(executor.map(process_one, tasks), total=len(tasks), desc='Preprocessing'))
        for in_name, out_name in results:
            print(f"Preprocessed: {in_name} -> {out_name}")

if __name__ == '__main__':
    main() 