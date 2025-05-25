import re
import json
import argparse
import os
from rapidfuzz import fuzz, process

# Define main headers and subheaders for hierarchy
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
    r"Halaman \\d+ dari \\d+ hal\." , # page numbers
    r"^Untuk Salinantera Pengadilan Agama Ban.*$", # remove this footer line
]

EXPECTED_SECTION_HEADERS = {"Preamble", "PEMOHON", "TERMOHON"}

def remove_boilerplate(text):
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.DOTALL)
    return text

def extract_preamble(text):
    # Extract lines before the first main header (now DUDUK PERKARA)
    lines = text.splitlines()
    preamble_lines = []
    duduk_perkara_found = False
    for i, line in enumerate(lines):
        if is_header(line, ["DUDUK PERKARA"]):
            duduk_perkara_found = True
            return "\n".join(preamble_lines).strip(), "\n".join(lines[i:]).strip()
        preamble_lines.append(line)
    return "\n".join(preamble_lines).strip(), ""

def is_header(line, header_list, threshold=85):
    match, score, _ = process.extractOne(line.strip(), header_list, scorer=fuzz.ratio)
    return match if score >= threshold else None

def group_sections_fuzzy(text):
    lines = text.splitlines()
    sections = []
    current_section = None
    current_subsection = None
    for line in lines:
        if not line.strip():
            continue
        # Special handling for 'Perincian biaya' even if it appears mid-line or with a colon
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
        # Add line to the right place
        if current_subsection:
            current_subsection["content"] += line + "\n"
        elif current_section:
            current_section["content"] += line + "\n"
    # Add last section/subsection
    if current_subsection and current_section:
        current_section["subsections"].append(current_subsection)
    if current_section:
        sections.append(current_section)
    return sections

def normalize_line(line):
    # Replace curly apostrophes with straight, collapse whitespace, and strip
    return re.sub(r"[’‘`´]", "'", re.sub(r"\s+", " ", line)).strip()

def print_missing_text(original_text, reconstructed_text):
    # Split both texts into lines for easier comparison
    orig_lines = [normalize_line(line) for line in original_text.splitlines() if line.strip()]
    recon_lines = [normalize_line(line) for line in reconstructed_text.splitlines() if line.strip()]
    # Remove expected section headers from extra lines
    missing = [line for line in orig_lines if line not in recon_lines]
    extra = [line for line in recon_lines if line not in orig_lines and line not in EXPECTED_SECTION_HEADERS]
    if missing:
        print("\nMISSING LINES (in original but not in output):")
        for line in missing[:20]:  # Show up to 20 missing lines
            print(f"- {line}")
        if len(missing) > 20:
            print(f"...and {len(missing)-20} more lines.")
    if extra:
        print("\nEXTRA LINES (in output but not in original):")
        for line in extra[:20]:
            print(f"- {line}")
        if len(extra) > 20:
            print(f"...and {len(extra)-20} more lines.")
    if not missing and not extra:
        print("No missing or extra lines. All lines match.")

def validate_coverage(original_text, sections):
    def flatten(sections):
        out = []
        for s in sections:
            out.append(s["header"] + "\n" + s["content"] + "\n")
            if "subsections" in s:
                for sub in s["subsections"]:
                    out.append(sub["header"] + "\n" + sub["content"] + "\n")
        return "".join(out)
    reconstructed = flatten(sections)
    # Normalize apostrophes and whitespace for robust comparison
    def norm(s):
        return re.sub(r"[’‘`´]", "'", re.sub(r"\s+", "", s))
    orig_comp = norm(original_text)
    recon_comp = norm(reconstructed)
    if orig_comp != recon_comp:
        print("WARNING: Not all text was covered by the sectioning logic!")
        import difflib
        diff = difflib.unified_diff(orig_comp, recon_comp, lineterm="")
        print("DIFF (first 1000 chars):\n" + "".join(list(diff)[:1000]))
        print_missing_text(original_text, reconstructed)
    else:
        print("Validation passed: All text is covered by the sections.")

def sections_to_markdown(sections):
    lines = []
    for s in sections:
        lines.append(f"# {s['header']}")
        if s["content"]:
            lines.append(s["content"].strip())
        if "subsections" in s:
            for sub in s["subsections"]:
                lines.append(f"## {sub['header']}")
                if sub["content"]:
                    lines.append(sub["content"].strip())
    return "\n\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structure cleaned legal text into hierarchical sections (fuzzy, robust).")
    parser.add_argument("input_file", help="Path to cleaned text file")
    parser.add_argument("-o", "--output", default="structured_sections.json", help="Output JSON file")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Remove boilerplate first!
    text = remove_boilerplate(text)
    print("[DEBUG] After boilerplate removal:\n", text[:1000], "\n---\n")

    preamble, rest = extract_preamble(text)
    print("[DEBUG] After preamble extraction:\n", preamble[:500], "\n---\n", rest[:500], "\n---\n")
    sections = []
    if preamble:
        sections.append({"header": "Preamble", "content": preamble})
    # No party extraction, just group sections from rest
    sections += group_sections_fuzzy(rest)
    validate_coverage(text, sections)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)
    md_path = os.path.splitext(args.output)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(sections_to_markdown(sections))
    print(f"Section structuring complete. Output saved to {args.output} and {md_path}")