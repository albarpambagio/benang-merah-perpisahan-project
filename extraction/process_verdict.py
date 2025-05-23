import subprocess
import argparse
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command: list) -> bool:
    """Helper function to run shell commands"""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(command)}")
        logger.error(e.stderr)
        return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Full processing pipeline for court verdicts')
    parser.add_argument('input_pdf', help='Path to input PDF file')
    parser.add_argument('--keep-cleaned', action='store_true', 
                       help='Keep intermediate cleaned text file')
    parser.add_argument('--output-dir', default='.', 
                       help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate filenames
    base_name = os.path.splitext(os.path.basename(args.input_pdf))[0]
    cleaned_file = os.path.join(args.output_dir, f"{base_name}_cleaned.txt")
    json_file = os.path.join(args.output_dir, f"{base_name}_structured_sections.json")
    
    # Step 1: Extract and clean text
    logger.info(f"Step 1/2: Extracting and cleaning text from {args.input_pdf}")
    if not run_command([
        sys.executable, 'extract_pdf.py', 
        args.input_pdf, 
        '-o', cleaned_file
    ]):
        sys.exit(1)
    
    # Step 2: Structure the cleaned text
    logger.info(f"Step 2/2: Structuring cleaned text into {json_file}")
    if not run_command([
        sys.executable, 'structure_output.py', 
        cleaned_file, 
        json_file
    ]):
        sys.exit(1)
    
    # Clean up intermediate file if requested
    if not args.keep_cleaned:
        try:
            os.remove(cleaned_file)
            logger.info(f"Removed intermediate file: {cleaned_file}")
        except OSError as e:
            logger.warning(f"Could not remove {cleaned_file}: {str(e)}")
    
    logger.info(f"✅ Processing complete. Final output saved to {json_file}")
    logger.info("Note: The output is now a list of sections (header/content), not a field-based dict.")

if __name__ == "__main__":
    main()