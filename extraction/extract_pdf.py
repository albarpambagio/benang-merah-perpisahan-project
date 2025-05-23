import fitz  # PyMuPDF
import re

def clean_text(raw_text):
    lines = raw_text.split('\n')
    cleaned_lines = []

    for line in lines:
        # Remove headers/footers/artifacts
        if re.match(r"^\s*(hkama|ahkamah|mah Agung|blik Indonesi|Direktori Putusan|Halaman \d+|Disclaimer|putusan\.mahkamahagung\.go\.id)", line):
            continue
        if re.search(r"Email : kepaniteraan@mahkamahagung\.go\.id", line):
            continue
        if line.strip() == "":
            continue

        cleaned_lines.append(line.strip())

    # Join broken lines (e.g., ending with "," or lowercase continuation)
    full_text = []
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


# Load and extract text from PDF
pdf_path = "1_Pdt.G_2025_PA.Bko.pdf"
doc = fitz.open(pdf_path)
raw_text = "\n".join(page.get_text("text") for page in doc)

# Clean the text
cleaned_text = clean_text(raw_text)

# Save to file
with open("cleaned_putusan_output.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print("Cleaned text extraction complete. Output saved to 'cleaned_putusan_output.txt'.")
