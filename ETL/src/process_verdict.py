import subprocess
import argparse
import os
import sys
import logging
from typing import List
from utils.file_utils import validate_file_exists, generate_output_paths

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command: List[str]) -> bool:
    """Helper function to run shell commands with robust error handling"""
    try:
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8'
        )
        logger.debug(result.stdout)
        if result.stderr:
            logger.debug(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(command)}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running command: {str(e)}")
        return False

def cleanup_intermediate_files(files: List[str]) -> None:
    """Remove intermediate files with error handling"""
    for file_path in files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed intermediate file: {file_path}")
        except OSError as e:
            logger.warning(f"Could not remove {file_path}: {str(e)}")

def main() -> None:
    """Main execution function for processing court verdict PDFs"""
    parser = argparse.ArgumentParser(
        description='Full processing pipeline for court verdict PDFs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_pdf', 
        help='Path to input PDF file'
    )
    parser.add_argument(
        '--output-dir', 
        default='output',
        help='Directory to save output files'
    )
    parser.add_argument(
        '--keep-intermediate', 
        action='store_true',
        help='Keep intermediate processing files'
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

    # Validate input file
    if not os.path.exists(args.input_pdf):
        logger.error(f"Input PDF not found: {args.input_pdf}")
        sys.exit(1)

    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    paths = generate_output_paths(args.input_pdf, args.output_dir)

    # Step 1: Extract and clean text from PDF
    logger.info("Step 1/3: Extracting and cleaning text from PDF")
    if not run_command([
        sys.executable, 
        os.path.join(os.path.dirname(__file__), 'extract_pdf.py'), 
        args.input_pdf, 
        '-o', paths['cleaned_txt']
    ]) or not validate_file_exists(paths['cleaned_txt']):
        sys.exit(1)

    if getattr(args, 'txt_output', False):
        logger.info(f"--txt-output set, exiting after cleaned text: {paths['cleaned_txt']}")
        sys.exit(0)

    # Step 2: Structure the cleaned text
    logger.info("Step 2/3: Structuring text into sections")
    if not run_command([
        sys.executable, 
        os.path.join(os.path.dirname(__file__), 'structure_output.py'),
        paths['cleaned_txt'],
        '-o', paths['structured_json']
    ]) or not validate_file_exists(paths['structured_json']):
        sys.exit(1)

    # Step 3: Preprocess the structured Markdown
    logger.info("Step 3/3: Preprocessing structured sections")
    if not run_command([
        sys.executable, 
        os.path.join(os.path.dirname(__file__), 'preprocess_sections.py'),
        paths['structured_md'],
        '-o', paths['preprocessed_md']
    ]) or not validate_file_exists(paths['preprocessed_md']):
        sys.exit(1)

    # Final output - rename the preprocessed file
    try:
        os.rename(paths['preprocessed_md'], paths['final_output'])
        logger.info(f"✅ Processing complete. Final output saved to {paths['final_output']}")
    except OSError as e:
        logger.error(f"Failed to create final output: {str(e)}")
        sys.exit(1)

    # Clean up intermediate files if not requested to keep
    if not args.keep_intermediate:
        cleanup_intermediate_files([
            paths['cleaned_txt'],
            paths['structured_json'],
            paths['structured_md'],
            paths['preprocessed_md']
        ])

if __name__ == "__main__":
    main()