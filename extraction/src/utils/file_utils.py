import os
from typing import Dict

def validate_file_exists(file_path: str) -> bool:
    """
    Check if a file exists and is not empty.
    """
    if not os.path.exists(file_path):
        return False
    if os.path.getsize(file_path) == 0:
        return False
    return True

def generate_output_paths(input_pdf: str, output_dir: str) -> Dict[str, str]:
    """
    Generate consistent output filenames based on input PDF.
    """
    base_name = os.path.splitext(os.path.basename(input_pdf))[0]
    return {
        'cleaned_txt': os.path.join(output_dir, f"{base_name}_cleaned.txt"),
        'structured_json': os.path.join(output_dir, f"{base_name}_structured.json"),
        'structured_md': os.path.join(output_dir, f"{base_name}_structured.md"),
        'preprocessed_md': os.path.join(output_dir, f"{base_name}_preprocessed.md"),
        'final_output': os.path.join(output_dir, f"{base_name}.md")
    } 