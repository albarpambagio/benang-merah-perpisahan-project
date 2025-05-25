#!/usr/bin/env python3
"""
Windows Parallel PDF Processor (Threaded Version):
- Uses ThreadPoolExecutor for parallelism (I/O-bound)
- Live progress bar with tqdm
- Error handling, deduplication, resume, validation, summary
"""

import argparse
import csv
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from utils.pdf_utils import validate_pdf, check_duplicates
from utils.state_utils import load_state, save_state, update_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parallel_pdf_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.processed_files = set()
        self.failed_files = set()
        self.state_file = 'processing_state.json'
        self.duplicates_file = 'duplicates_found.txt'
        self.summary_file = 'processing_summary.csv'
        self.max_retries = 3
        self.min_pdf_size = 1024  # 1KB minimum size to process

    def load_state(self) -> None:
        load_state(self)

    def save_state(self) -> None:
        save_state(self)

    def update_state(self, file_path: str, success: bool) -> None:
        update_state(self, file_path, success)

    def find_pdfs(self, input_path: str) -> List[Tuple[str, int]]:
        pdfs = []
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    full_path = os.path.join(root, file)
                    try:
                        size = os.path.getsize(full_path)
                        if size >= self.min_pdf_size:
                            pdfs.append((full_path, size))
                    except OSError:
                        continue
        return sorted(pdfs, key=lambda x: x[1], reverse=True)

    def check_duplicates(self, pdfs: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        return check_duplicates(pdfs, self.duplicates_file)

    def validate_pdf(self, pdf_path: str) -> bool:
        return validate_pdf(pdf_path, self.min_pdf_size)

    def process_single_pdf(self, pdf_path: str, output_dir: str, keep_intermediate: bool = False, txt_output: bool = False) -> bool:
        import subprocess
        process_verdict_path = os.path.join(os.path.dirname(__file__), 'process_verdict.py')
        cmd = [
            sys.executable,
            process_verdict_path,
            pdf_path,
            '--output-dir', output_dir,
        ]
        if keep_intermediate:
            cmd.append('--keep-intermediate')
        if txt_output:
            cmd.append('--txt-output')
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            )
            logger.debug(f"Successfully processed {pdf_path}")
            logger.debug(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to process {pdf_path}")
            logger.error(e.stderr)
            return False
        except Exception as e:
            logger.error(f"Unexpected error processing {pdf_path}: {str(e)}")
            return False

    def worker(self, queue: Queue, output_dir: str, keep_intermediate: bool, pbar: tqdm, txt_output: bool = False):
        while True:
            try:
                pdf_path = queue.get_nowait()
            except Empty:
                break
            success = False
            for attempt in range(self.max_retries):
                if attempt > 0:
                    logger.info(f"Retry {attempt} for {pdf_path}")
                success = self.process_single_pdf(pdf_path, output_dir, keep_intermediate, txt_output)
                if success:
                    break
            self.update_state(pdf_path, success)
            pbar.update(1)
            queue.task_done()

    def run_parallel_processing(self, pdf_list: List[str], output_dir: str, jobs: int = 4, keep_intermediate: bool = False, resume: bool = True, dry_run: bool = False, txt_output: bool = False) -> None:
        if not pdf_list:
            logger.warning("No PDFs to process")
            return
        logger.info(f"Starting parallel processing of {len(pdf_list)} PDFs with {jobs} workers")
        if dry_run:
            logger.info("Dry run - would process:")
            for pdf in pdf_list:
                logger.info(f"  - {pdf}")
            return
        queue = Queue()
        for pdf in pdf_list:
            queue.put(pdf)
        with tqdm(total=len(pdf_list)) as pbar:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = [executor.submit(self.worker, queue, output_dir, keep_intermediate, pbar, txt_output) for _ in range(jobs)]
                queue.join()
        logger.info("Parallel processing completed")

    def generate_summary(self) -> None:
        total = len(self.processed_files) + len(self.failed_files)
        success_rate = (len(self.processed_files) / total * 100) if total > 0 else 0
        with open(self.summary_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total PDFs processed', total])
            writer.writerow(['Successfully processed', len(self.processed_files)])
            writer.writerow(['Failed to process', len(self.failed_files)])
            writer.writerow(['Success rate', f"{success_rate:.2f}%"])
        logger.info(f"Processing summary saved to {self.summary_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Windows Parallel PDF Processor (Threaded Version)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_path',
        help='Path to input directory containing PDFs or single PDF file',
        nargs='?',
        default=None
    )
    parser.add_argument(
        '--output-dir',
        help='Directory to save output files',
        default='output'
    )
    parser.add_argument(
        '--jobs',
        type=int,
        help='Number of parallel workers to use',
        default=4
    )
    parser.add_argument(
        '--keep-intermediate',
        action='store_true',
        help='Keep intermediate processing files'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume previous processing (skip already processed files)'
    )
    parser.add_argument(
        '--no-dedupe',
        action='store_true',
        help='Skip duplicate detection'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually processing'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--txt-output',
        action='store_true',
        help='Only output cleaned, preprocessed text as .txt (no markdown or JSON)'
    )
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    processor = PDFProcessor()
    processor.load_state()

    if not args.input_path:
        logger.error("Input path is required")
        sys.exit(1)

    # Find PDF files
    if os.path.isfile(args.input_path) and args.input_path.lower().endswith('.pdf'):
        pdfs = [(args.input_path, os.path.getsize(args.input_path))]
    else:
        pdfs = processor.find_pdfs(args.input_path)

    if not pdfs:
        logger.error("No valid PDF files found")
        sys.exit(1)

    logger.info(f"Found {len(pdfs)} PDF files")

    # Check for duplicates
    if not args.no_dedupe:
        pdfs = processor.check_duplicates(pdfs)
        logger.info(f"Processing {len(pdfs)} unique PDF files after deduplication")

    # Filter out already processed files if resuming
    if args.resume:
        pdfs = [(p, s) for p, s in pdfs if p not in processor.processed_files]
        logger.info(f"Resuming - {len(pdfs)} PDFs remaining to process")

    # Validate PDFs
    valid_pdfs = []
    for path, size in pdfs:
        if processor.validate_pdf(path):
            valid_pdfs.append(path)
        else:
            logger.warning(f"Skipping invalid PDF: {path}")
            processor.failed_files.add(path)
    
    # Log files that are not in valid_pdfs (deduplication or other skips)
    all_pdf_paths = set([p for p, s in pdfs])
    valid_pdf_set = set(valid_pdfs)
    skipped = all_pdf_paths - valid_pdf_set
    for skipped_file in skipped:
        logger.info(f"File not processed (skipped or duplicate): {skipped_file}")

    if not valid_pdfs:
        logger.error("No valid PDFs to process after validation")
        sys.exit(1)

    # Run parallel processing
    processor.run_parallel_processing(
        valid_pdfs,
        args.output_dir,
        args.jobs,
        args.keep_intermediate,
        args.resume,
        args.dry_run,
        args.txt_output
    )

    # If --txt-output, run batch post-processing on all cleaned text files
    if args.txt_output:
        import subprocess
        # Run post-processing, output directly to output_dir
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "preprocess_cleaned_txt.py"),
            "--input-dir", args.output_dir,
            "--output-dir", args.output_dir
        ]
        print(f"Running batch text post-processing: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error("Batch text post-processing failed.")
        else:
            logger.info(f"Batch text post-processing complete. Preprocessed .txt files are in {args.output_dir}")
            # Delete all _cleaned.txt files from output_dir
            for fname in os.listdir(args.output_dir):
                if fname.endswith('_cleaned.txt'):
                    try:
                        os.remove(os.path.join(args.output_dir, fname))
                        logger.info(f"Removed intermediate file: {fname}")
                    except Exception as e:
                        logger.warning(f"Could not remove {fname}: {str(e)}")

    # Save state and generate summary
    processor.save_state()
    processor.generate_summary()

    # Print missing files for user
    import glob
    input_dir = args.input_path if os.path.isdir(args.input_path) else os.path.dirname(args.input_path)
    all_pdfs = set(glob.glob(os.path.join(input_dir, '*.pdf')))
    processed = processor.processed_files
    failed = processor.failed_files
    missing = all_pdfs - processed - failed
    if missing:
        print("\nFiles not processed (not in processed or failed):")
        for f in missing:
            print(f"- {f}")
        print("Possible reasons: deduplication, validation failure, or script interruption.")
    else:
        print("\nAll files were processed or failed and are accounted for.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)