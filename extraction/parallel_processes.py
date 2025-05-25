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
        """Load processing state from previous run"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.processed_files = set(state.get('processed', []))
                    self.failed_files = set(state.get('failed', []))
                logger.info(f"Loaded state: {len(self.processed_files)} processed, {len(self.failed_files)} failed")
        except Exception as e:
            logger.warning(f"Could not load state file: {str(e)}")

    def save_state(self) -> None:
        """Save current processing state"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    'processed': list(self.processed_files),
                    'failed': list(self.failed_files),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")

    def update_state(self, file_path: str, success: bool) -> None:
        if success:
            self.processed_files.add(file_path)
            if file_path in self.failed_files:
                self.failed_files.remove(file_path)
        else:
            self.failed_files.add(file_path)
        self.save_state()

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
        size_map = defaultdict(list)
        unique_pdfs = []
        duplicates_found = 0
        for path, size in pdfs:
            filename = os.path.basename(path)
            size_map[(filename, size)].append(path)
        for (filename, size), paths in size_map.items():
            if len(paths) > 1:
                duplicates_found += len(paths) - 1
                with open(self.duplicates_file, 'a') as f:
                    f.write(f"Duplicate: {filename} ({size} bytes)\n")
                    for p in paths:
                        f.write(f"  - {p}\n")
                unique_pdfs.append((paths[0], size))
            else:
                unique_pdfs.append((paths[0], size))
        if duplicates_found:
            logger.warning(f"Found {duplicates_found} duplicate PDFs. See {self.duplicates_file}")
        return unique_pdfs

    def validate_pdf(self, pdf_path: str) -> bool:
        try:
            if not os.path.exists(pdf_path):
                logger.debug(f"File not found: {pdf_path}")
                return False
            size = os.path.getsize(pdf_path)
            if size < self.min_pdf_size:
                logger.debug(f"File too small ({size} bytes): {pdf_path}")
                return False
            with open(pdf_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    logger.debug(f"Invalid PDF header in: {pdf_path}")
                    return False
            return True
        except Exception as e:
            logger.debug(f"Validation failed for {pdf_path}: {str(e)}")
            return False

    def process_single_pdf(self, pdf_path: str, output_dir: str, keep_intermediate: bool = False, txt_output: bool = False) -> bool:
        import subprocess
        process_verdict_path = os.path.join(os.path.dirname(__file__), 'src', 'process_verdict.py')
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