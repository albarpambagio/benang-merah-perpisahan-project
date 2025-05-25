import fitz  # PyMuPDF
import re
import os
from typing import Optional, List, Tuple

def extract_pdf_text(pdf_path: str) -> Optional[str]:
    """
    Extract text from a PDF file using PyMuPDF.
    Returns the extracted text as a single string, or None if extraction fails.
    """
    try:
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text("text") for page in doc)
    except Exception:
        return None

def clean_text(raw_text: str) -> str:
    """
    Clean extracted PDF text by:
    1. Removing headers/footers/artifacts
    2. Joining broken lines (those ending with punctuation or starting with lowercase)
    """
    remove_patterns = [
        r"^\s*(hkama|ahkamah|mah Agung|blik Indonesi|Direktori Putusan|Halaman \d+|Disclaimer|putusan\.mahkamahagung\.go\.id)",
        r"Email : kepaniteraan@mahkamahagung\.go\.id"
    ]
    cleaned_lines: List[str] = []
    for line in raw_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if any(re.search(pattern, line) for pattern in remove_patterns):
            continue
        cleaned_lines.append(line)
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

def validate_pdf(pdf_path: str, min_pdf_size: int = 1024) -> bool:
    """
    Validate that a file is a PDF and meets minimum size.
    """
    try:
        if not os.path.exists(pdf_path):
            return False
        size = os.path.getsize(pdf_path)
        if size < min_pdf_size:
            return False
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                return False
        return True
    except Exception:
        return False

def check_duplicates(pdfs: List[Tuple[str, int]], duplicates_file: str) -> List[Tuple[str, int]]:
    """
    Detect duplicate PDFs by filename and size. Log duplicates to file. Return unique PDFs.
    """
    from collections import defaultdict
    size_map = defaultdict(list)
    unique_pdfs = []
    duplicates_found = 0
    for path, size in pdfs:
        filename = os.path.basename(path)
        size_map[(filename, size)].append(path)
    for (filename, size), paths in size_map.items():
        if len(paths) > 1:
            duplicates_found += len(paths) - 1
            with open(duplicates_file, 'a') as f:
                f.write(f"Duplicate: {filename} ({size} bytes)\n")
                for p in paths:
                    f.write(f"  - {p}\n")
            unique_pdfs.append((paths[0], size))
        else:
            unique_pdfs.append((paths[0], size))
    return unique_pdfs 