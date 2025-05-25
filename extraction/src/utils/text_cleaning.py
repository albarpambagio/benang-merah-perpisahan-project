import re
from typing import List

def remove_boilerplate(text: str, patterns: List[str]) -> str:
    """
    Remove boilerplate patterns from text using provided regex patterns.
    """
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.MULTILINE)
    return text

def normalize_whitespace(text: str) -> str:
    """
    Remove extra blank lines and trim spaces from each line.
    """
    lines = [line.strip() for line in text.splitlines()]
    cleaned = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    return "\n".join(cleaned).strip()

def clean_whitespace(text: str) -> str:
    """
    Remove leading/trailing spaces and collapse multiple blank lines to one.
    """
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def remove_artifacts(text: str) -> str:
    """
    Remove lines that are just numbers or page markers.
    """
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Page \d+.*$", "", text, flags=re.MULTILINE)
    return text 