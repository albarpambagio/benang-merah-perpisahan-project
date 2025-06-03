import json
import os
from pathlib import Path
from PyPDF2 import PdfReader
from tqdm import tqdm

def count_pdf_pages(pdf_path):
    """Count the number of pages in a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            return len(pdf.pages)
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return None

def main():
    # Paths
    pdfs_dir = Path("data/pdfs")
    input_jsonl = Path("data/court_cases_output.jsonl")
    output_jsonl = Path("data/court_cases_output_with_pages.jsonl")
    
    # Get list of PDF files
    pdf_files = {f.name: f for f in pdfs_dir.glob("*.pdf")}
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process JSONL file
    total_processed = 0
    total_updated = 0
    
    with open(input_jsonl, 'r', encoding='utf-8') as infile, \
         open(output_jsonl, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, desc="Processing cases"):
            case = json.loads(line)
            total_processed += 1
            
            # Get PDF filename from metadata
            pdf_filename = case.get('metadata', {}).get('pdf_filename')
            if pdf_filename and pdf_filename in pdf_files:
                # Count pages
                page_count = count_pdf_pages(pdf_files[pdf_filename])
                if page_count is not None:
                    # Update metadata
                    if 'metadata' not in case:
                        case['metadata'] = {}
                    case['metadata']['page_count'] = page_count
                    total_updated += 1
            
            # Write updated case to output file
            outfile.write(json.dumps(case, ensure_ascii=False) + '\n')
    
    print(f"\nProcessing complete!")
    print(f"Total cases processed: {total_processed}")
    print(f"Cases updated with page count: {total_updated}")

if __name__ == "__main__":
    main() 