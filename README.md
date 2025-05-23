Get-ChildItem -Path scrapy\pdfs -Filter *.pdf | Measure-Object | % { $_.Count }



Usage Examples:

Complete processing (recommended):

python process_verdict.py input.pdf --output-dir ./results

Individual steps (for debugging):

# Step 1: Extract text
python extract_pdf.py input.pdf -o cleaned.txt

# Step 2: Structure text
python structure_output.py cleaned.txt -o output.json

Keep intermediate files:

python process_verdict.py input.pdf --keep-cleaned