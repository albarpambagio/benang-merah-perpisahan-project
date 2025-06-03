# Extraction Pipeline

This directory contains scripts and utilities for extracting, cleaning, and structuring Indonesian court verdict PDFs into analyzable text and markdown files.

## Main Scripts (in `src/`)

- **extract_pdf.py**  
  Extracts and cleans text from a PDF file.  
  **Arguments:**  
  - `input_pdf` (positional): Path to input PDF file  
  - `-o`, `--output`: Path to output text file (default: `cleaned_putusan_output.txt`)

- **structure_output.py**  
  Structures cleaned text into hierarchical sections (JSON/Markdown).  
  **Arguments:**  
  - `input_file` (positional): Path to cleaned text file  
  - `-o`, `--output`: Output JSON file (default: `structured_sections.json`)

- **preprocess_sections.py**  
  Further cleans and standardizes sectioned text (e.g., whitespace, headers).  
  **Arguments:**  
  - `input_file` (positional): Path to input file (e.g., structured_sections.md)  
  - `-o`, `--output`: Output cleaned file (default: `cleaned_sections.md`)

- **preprocess_cleaned_txt.py**  
  Batch preprocesses all cleaned text files in a folder (removes boilerplate, normalizes whitespace).  
  **Arguments:**  
  - `--input-dir`: Directory containing cleaned `.txt` files (**required**)  
  - `--output-dir`: Directory to save preprocessed `.txt` files (**required**)  
  - `--jobs`: Number of parallel workers (default: 4)

- **process_verdict.py**  
  Runs the full pipeline for a single PDF (extraction → structuring → cleaning). Handles intermediate files.  
  **Arguments:**  
  - `input_pdf` (positional): Path to input PDF file  
  - `--output-dir`: Directory to save output files (default: `output`)  
  - `--keep-intermediate`: Keep intermediate processing files  
  - `--debug`: Enable debug logging  
  - `--txt-output`: Only output cleaned, preprocessed text as `.txt` (no markdown or JSON)

- **parallel_processes.py**  
  Batch processes multiple PDFs in parallel, with deduplication and state management.  
  **When using `--txt-output`, this script will also run `preprocess_cleaned_txt.py` on the output directory. The final preprocessed `.txt` files will be written directly to the output directory, and all intermediate `_cleaned.txt` files will be deleted.**  
  **Arguments:**  
  - `input_path` (positional): Path to input directory containing PDFs or single PDF file  
  - `--output-dir`: Directory to save output files (default: `output`)  
  - `--jobs`: Number of parallel workers to use (default: 4)  
  - `--keep-intermediate`: Keep intermediate processing files  
  - `--resume`: Resume previous processing (skip already processed files)  
  - `--no-dedupe`: Skip duplicate detection  
  - `--dry-run`: Show what would be done without actually processing  
  - `--debug`: Enable debug logging  
  - `--txt-output`: Only output cleaned, preprocessed text as `.txt` (no markdown or JSON). **Final preprocessed files will be in your output dir, with no _cleaned.txt files remaining.**

## Utility Modules (in `src/utils/`)

- **pdf_utils.py**: PDF extraction, cleaning, validation, duplicate detection.
- **text_cleaning.py**: Boilerplate removal, whitespace/artifact cleaning.
- **text_structure.py**: Sectioning, fuzzy header logic, preamble extraction.
- **file_utils.py**: File validation and output path generation.
- **state_utils.py**: State management for parallel processing.

## Common Usage

### 1. Full Pipeline (Single PDF)

```bash
python src/process_verdict.py path/to/input.pdf --output-dir ./results
```

### 2. Step-by-Step (Debugging or Custom Flow)

```bash
# Step 1: Extract and clean text
python src/extract_pdf.py path/to/input.pdf -o cleaned.txt

# Step 2: Structure text into sections
python src/structure_output.py cleaned.txt -o output.json

# Step 3: Further clean/standardize sections
python src/preprocess_sections.py output.json -o cleaned_sections.md
```

### 3. Batch Processing (All PDFs in a Folder)

```bash
python src/parallel_processes.py path/to/pdf_folder --output-dir ./results --jobs 4
```

### 4. Batch Processing for Cleaned TXT Output Only

```bash
python src/parallel_processes.py path/to/pdf_folder --output-dir ./results --jobs 4 --txt-output
# After processing, final preprocessed .txt files will be in ./results/ (no _cleaned.txt files will remain)
```

### 5. Keep Intermediate Files

```bash
python src/process_verdict.py path/to/input.pdf --output-dir ./results --keep-intermediate
```

### 6. Preprocess All Cleaned Texts in a Folder

```bash
python src/preprocess_cleaned_txt.py --input-dir ./cleaned_txts --output-dir ./preprocessed_txts
```

---

- All scripts support `-h` or `--help` for more options.
- Utility modules can be imported for custom workflows.
- See comments and docstrings in each file for details. 