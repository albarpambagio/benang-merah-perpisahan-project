import argparse
import os
from utils.text_cleaning import remove_boilerplate, remove_artifacts, clean_whitespace
from typing import List

# Patterns for boilerplate and repetitive notices
BOILERPLATE_PATTERNS: List[str] = [
    r"Namun dalam hal-hal tertentu masih dimungkinkan terjadi permasalahan teknis terkait dengan akurasi dan keterkinian informasi yang kami sajikan, hal mana akan terus kami perbaiki dari waktu kewaktu\.?",
    r"Halaman \\d+ dari \\d+ hal\.?",
    r"^Untuk Salinantera Pengadilan Agama Ban.*$",
]

# Main section headers (add more as needed)
MAIN_HEADERS = [
    "DUDUK PERKARA",
    "PERTIMBANGAN HUKUM",
    "Mengadili",
    "Penutup",
    "Perincian biaya",
]

import re
HEADER_PATTERN = re.compile(r"^(%s)[\s:：-]*$" % "|".join([re.escape(h) for h in MAIN_HEADERS]), re.IGNORECASE)

def standardize_headers(text):
    def repl(match):
        return f"\n{match.group(1).upper()}\n"
    # Place headers on their own line, uppercase
    return HEADER_PATTERN.sub(repl, text)

def fix_broken_lines(text):
    lines = text.splitlines()
    fixed_lines = []
    buffer = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                fixed_lines.append(buffer)
                buffer = ""
            fixed_lines.append("")
            continue
        if HEADER_PATTERN.match(stripped):
            if buffer:
                fixed_lines.append(buffer)
                buffer = ""
            fixed_lines.append(stripped)
        else:
            if buffer:
                buffer += " " + stripped
            else:
                buffer = stripped
    if buffer:
        fixed_lines.append(buffer)
    return "\n".join(fixed_lines)

def preprocess_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = remove_boilerplate(text, BOILERPLATE_PATTERNS)
    text = remove_artifacts(text)
    text = standardize_headers(text)
    text = fix_broken_lines(text)
    text = clean_whitespace(text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Preprocessing complete. Cleaned file saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess legal document for sectioning.")
    parser.add_argument("input_file", help="Path to input file (e.g., structured_sections.md)")
    parser.add_argument("-o", "--output", default="cleaned_sections.md", help="Output cleaned file")
    args = parser.parse_args()
    preprocess_file(args.input_file, args.output) 