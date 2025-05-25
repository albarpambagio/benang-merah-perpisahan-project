import re
from rapidfuzz import fuzz, process
from typing import List, Dict, Any, Tuple, Optional

MAIN_HEADERS = [
    "PEMOHON",
    "TERMOHON",
    "DUDUK PERKARA",
    "PERTIMBANGAN HUKUM",
    "Mengadili",
    "Penutup",
    "Perincian biaya",
]
SUBHEADERS_UNDER = {
    "PERTIMBANGAN HUKUM": [
        "Pokok Perkara",
        "Analisis Pembuktian",
        "Fakta Hukum",
        "Pertimbangan Petitum Perceraian",
        "Pertimbangan ex officio tentang Akibat Putusnya Perkawinan",
        "Pertimbangan Petitum Mut'ah dan Nafkah Iddah",
        "Biaya Perkara",
    ]
}
ALL_SUBHEADERS = [sh for subs in SUBHEADERS_UNDER.values() for sh in subs]
ALL_HEADERS = MAIN_HEADERS + ALL_SUBHEADERS

BOILERPLATE_PATTERNS = [
    r"Kepaniteraan Mahkamah Agung Republik Indonesia[\s\S]*?fungsi peradilan\." ,
    r"Dalam hal Anda menemukan inakurasi informasi[\s\S]*?harap segera hubungi Kepaniteraan Mahkamah Agung RI melalui :",
    r"Halaman \\d+ dari \\d+ hal\." ,
    r"^Untuk Salinantera Pengadilan Agama Ban.*$",
]

EXPECTED_SECTION_HEADERS = {"Preamble", "PEMOHON", "TERMOHON"}

def remove_boilerplate(text: str) -> str:
    """
    Remove boilerplate patterns from text using internal patterns.
    """
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.DOTALL)
    return text

def extract_preamble(text: str) -> Tuple[str, str]:
    """
    Extract lines before the first main header (DUDUK PERKARA).
    Returns (preamble, rest_of_text).
    """
    lines = text.splitlines()
    preamble_lines = []
    for i, line in enumerate(lines):
        if is_header(line, ["DUDUK PERKARA"]):
            return "\n".join(preamble_lines).strip(), "\n".join(lines[i:]).strip()
        preamble_lines.append(line)
    return "\n".join(preamble_lines).strip(), ""

def is_header(line: str, header_list: List[str], threshold: int = 85) -> Optional[str]:
    """
    Fuzzy match a line to a list of headers. Returns the matched header or None.
    """
    match, score, _ = process.extractOne(line.strip(), header_list, scorer=fuzz.ratio)
    return match if score >= threshold else None

def group_sections_fuzzy(text: str) -> List[Dict[str, Any]]:
    """
    Group text into sections and subsections using fuzzy header matching.
    Returns a list of section dicts.
    """
    lines = text.splitlines()
    sections = []
    current_section = None
    current_subsection = None
    for line in lines:
        if not line.strip():
            continue
        if re.match(r"^Perincian biaya\s*[:\-]?", line.strip(), re.IGNORECASE):
            if current_section:
                if current_subsection:
                    current_section["subsections"].append(current_subsection)
                    current_subsection = None
                sections.append(current_section)
            current_section = {"header": "Perincian biaya", "content": line.strip(), "subsections": []}
            continue
        main_header = is_header(line, MAIN_HEADERS)
        if main_header and main_header != "Perincian biaya":
            if current_section:
                if current_subsection:
                    current_section["subsections"].append(current_subsection)
                    current_subsection = None
                sections.append(current_section)
            current_section = {"header": main_header, "content": "", "subsections": []}
            continue
        subheader = is_header(line, ALL_SUBHEADERS)
        if subheader and current_section and current_section["header"] == "PERTIMBANGAN HUKUM":
            if current_subsection:
                current_section["subsections"].append(current_subsection)
            current_subsection = {"header": subheader, "content": ""}
            continue
        if current_subsection:
            current_subsection["content"] += line + "\n"
        elif current_section:
            current_section["content"] += line + "\n"
    if current_subsection and current_section:
        current_section["subsections"].append(current_subsection)
    if current_section:
        sections.append(current_section)
    return sections 