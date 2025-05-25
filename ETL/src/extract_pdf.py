import argparse
import logging
from utils.pdf_utils import extract_pdf_text, clean_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function with argument parsing"""
    parser = argparse.ArgumentParser(description='Extract and clean text from court decision PDFs')
    parser.add_argument('input_pdf', help='Path to input PDF file')
    parser.add_argument('-o', '--output', help='Path to output text file', 
                       default='cleaned_putusan_output.txt')
    args = parser.parse_args()

    # Extract and clean text
    raw_text = extract_pdf_text(args.input_pdf)
    if not raw_text:
        logger.error("Failed to extract text from PDF")
        return
    cleaned_text = clean_text(raw_text)

    # Save output
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        logger.info(f"Successfully saved cleaned text to {args.output}")
    except IOError as e:
        logger.error(f"Failed to write output file: {str(e)}")

if __name__ == "__main__":
    main()