import fitz  # PyMuPDF
import re
import logging
import argparse
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_text(raw_text: str) -> str:
    """
    Clean extracted PDF text by:
    1. Removing headers/footers/artifacts
    2. Joining broken lines (those ending with punctuation or starting with lowercase)
    """
    # Patterns to remove (headers/footers/artifacts)
    remove_patterns = [
        r"^\s*(hkama|ahkamah|mah Agung|blik Indonesi|Direktori Putusan|Halaman \d+|Disclaimer|putusan\.mahkamahagung\.go\.id)",
        r"Email : kepaniteraan@mahkamahagung\.go\.id"
    ]
    
    # Process each line
    cleaned_lines: List[str] = []
    for line in raw_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if any(re.search(pattern, line) for pattern in remove_patterns):
            continue
        cleaned_lines.append(line)
    
    # Join broken lines
    full_text: List[str] = []
    buffer = ""
    for line in cleaned_lines:
        if buffer and (line[0].islower() or line[0] in '.,;:-'):
            buffer += " " + line
        else:
            if buffer:
                full_text.append(buffer)
            buffer = line
    if buffer:
        full_text.append(buffer)
    
    return "\n".join(full_text)

def extract_pdf_text(pdf_path: str) -> Optional[str]:
    """Extract text from PDF file with error handling"""
    try:
        logger.info(f"Extracting text from {pdf_path}")
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return None

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