import fitz  # PyMuPDF
import json
import re


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc])


def find_regex(pattern, text, group=1, default=None, flags=0):
    match = re.search(pattern, text, flags)
    return match.group(group).strip() if match else default


def extract_court_name(text):
    return find_regex(r"(PENGADILAN [A-Z\s]+)", text)


def extract_case_number(text):
    return find_regex(r"PUTUSAN\s+Nomor\s+([\d\/A-Za-z\.]+)", text)


def extract_decision_date(text):
    return find_regex(r"pada hari\s+\w+\s+tanggal\s+(\d{1,2} \w+ \d{4})", text, flags=re.IGNORECASE)


def extract_party(text, label):
    block = re.search(fr"{label},(.+?)(?:melawan|Kepaniteraan)", text, re.DOTALL | re.IGNORECASE)
    if not block:
        return {}

    content = block.group(1)
    return {
        "tanggal_lahir": find_regex(r"lahir di .*?,\s*(\d{2} \w+ \d{4})", content),
        "alamat": find_regex(r"kediaman di (.*?)(?:,|dalam hal|Provinsi)", content),
        "email": find_regex(r"email\s+(\S+@\S+)", content)
    }


def extract_children(text):
    children = re.findall(
        r"ANAK\s+[IVXL]*,\s+yang lahir di\s+(.*?)\s+pada tanggal\s+(\d{2} \w+ \d{4}),.*?Jenis Kelamin\s+(.*?)\.",
        text
    )
    return [
        {
            "tempat_lahir": loc.strip(),
            "tanggal_lahir": date.strip(),
            "jenis_kelamin": gender.strip().capitalize()
        }
        for loc, date, gender in children
    ]


def extract_divorce_reason(text):
    reason = re.search(r"DUDUK PERKARA(.+?)Berdasarkan alasan-alasan", text, re.DOTALL | re.IGNORECASE)
    return ' '.join(reason.group(1).strip().splitlines()) if reason else ""


def extract_witnesses(text):
    matches = re.findall(
        r"SAKSI\s+[IVXL]*.*?umur.*?(\d+).*?agama\s+(.*?)\,.*?pekerjaan.*?,.*?Provinsi\s+(.*?)\..*?kenal.*?(Bahwa.*?)Bahwa",
        text, re.DOTALL
    )
    return [
        {
            "umur": age,
            "agama": religion,
            "provinsi": province,
            "kesaksian": statement.strip()
        }
        for age, religion, province, statement in matches
    ]


def extract_verdict(text):
    section = re.search(r"Amar Putusan\s+MENGADILI:(.+?)Penutup", text, re.DOTALL | re.IGNORECASE)
    if section:
        body = section.group(1)
        mutah = find_regex(r"Mut[’']?ah.*?Rp[\s:]?([\d\.]+)", body)
        iddah = find_regex(r"Iddah.*?Rp[\s:]?([\d\.]+)", body)
        biaya = find_regex(r"biaya.*?Rp[\s:]?([\d\.]+)", body)

        return {
            "talak": "Talak satu raj’i dikabulkan",
            "mutah": int(mutah.replace(".", "")) if mutah else None,
            "nafkah_iddah": int(iddah.replace(".", "")) if iddah else None,
            "biaya_perkara": int(biaya.replace(".", "")) if biaya else None,
            "dibebankan_kepada": "Pemohon"
        }
    return {}


def extract_officer_name(text, role_label):
    return find_regex(rf"{role_label},\s*dto\.\s*(.+)", text)


def structure_verdict(text):
    return {
        "nomor_perkara": extract_case_number(text),
        "pengadilan": extract_court_name(text),
        "tanggal_putusan": extract_decision_date(text),
        "para_pihak": {
            "pemohon": extract_party(text, "PEMOHON"),
            "termohon": extract_party(text, "TERMOHON")
        },
        "anak": extract_children(text),
        "alasan_perceraian": extract_divorce_reason(text),
        "saksi": extract_witnesses(text),
        "amar_putusan": extract_verdict(text),
        "hakim": extract_officer_name(text, "Hakim Tunggal"),
        "panitera": extract_officer_name(text, "Panitera")
    }


# ==== MAIN PROGRAM ====
if __name__ == "__main__":
    pdf_file = "1_Pdt.G_2025_PA.Bko.pdf"
    raw_text = extract_text_from_pdf(pdf_file)

    structured_data = structure_verdict(raw_text)

    json_output_file = pdf_file.replace(".pdf", "_terstruktur.json")
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Output JSON saved to: {json_output_file}")
