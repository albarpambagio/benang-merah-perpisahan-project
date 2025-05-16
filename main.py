import requests
from bs4 import BeautifulSoup
import os
import json
import time

BASE_URL = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/perceraian.html"
OUTPUT_JSON = "output.jsonl"  # JSON Lines format
PDF_DIR = "pdfs"
MAX_PAGES = 1  # Change as needed

os.makedirs(PDF_DIR, exist_ok=True)

session = requests.Session()

# Helper to parse Indonesian date to YYYY-MM-DD
MONTHS = {
    'Januari': '01', 'Februari': '02', 'Maret': '03', 'April': '04',
    'Mei': '05', 'Juni': '06', 'Juli': '07', 'Agustus': '08',
    'September': '09', 'Oktober': '10', 'November': '11', 'Desember': '12'
}
def parse_date(date_str):
    if not date_str:
        return None
    parts = date_str.strip().split()
    if len(parts) == 3:
        day, month, year = parts
        month = MONTHS.get(month, '01')
        return f"{year}-{month}-{int(day):02d}"
    return date_str

def get_soup(url):
    resp = session.get(url)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def download_pdf(pdf_url, filename):
    path = os.path.join(PDF_DIR, filename)
    if os.path.exists(path):
        return path
    r = session.get(pdf_url, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)
    return path

def parse_case(case_url):
    soup = get_soup(case_url)
    item = {}
    # Metadata table
    table = soup.select_one('div.tab-content.ui-tabs-panel table.table')
    if table:
        for row in table.select('tr'):
            tds = row.select('td')
            if len(tds) < 2:
                continue
            label = tds[0].get_text(strip=True)
            value = tds[1].get_text(strip=True)
            if label == 'Nomor':
                item['case_number'] = value
            elif label == 'Tingkat Proses':
                item['process_level'] = value
            elif label == 'Klasifikasi':
                item['classification'] = ' '.join(a.get_text(strip=True) for a in tds[1].select('a'))
            elif label == 'Kata Kunci':
                item['keywords'] = value
            elif label == 'Tahun':
                item['year'] = value
            elif label == 'Tanggal Register':
                item['register_date'] = value
            elif label == 'Lembaga Peradilan':
                item['court_name'] = value
            elif label == 'Jenis Lembaga Peradilan':
                item['court_type'] = value
            elif label == 'Hakim Ketua':
                item['chief_judge'] = value
            elif label == 'Hakim Anggota':
                item['member_judges'] = value
            elif label == 'Panitera':
                item['panitera'] = value
            elif label == 'Amar':
                item['amar'] = value
            elif label == 'Amar Lainnya':
                item['amar_lainnya'] = value
            elif label == 'Catatan Amar':
                item['catatan_amar'] = tds[1].get_text(" ", strip=True)
            elif label == 'Tanggal Musyawarah':
                item['deliberation_date'] = value
            elif label == 'Tanggal Dibacakan':
                item['decision_date'] = value
            elif label == 'Kaidah':
                item['kaidah'] = value
            elif label == 'Abstrak':
                item['abstrak'] = value
    # PDF link
    pdf_url = None
    pdf_filename = None
    for card in soup.select('div.card.bg-success.mb-3'):
        for span in card.select('span'):
            icon = span.find('i', class_='icon-files')
            if icon and 'Download PDF' in span.get_text(strip=True):
                for a in card.select('a'):
                    link_text = a.get_text(strip=True)
                    if link_text.endswith('.pdf'):
                        pdf_url = a.get('href')
                        pdf_filename = link_text
                        break
                if pdf_url:
                    break
        if pdf_url:
            break
    if pdf_url and pdf_filename:
        item['pdf_url'] = pdf_url
        item['pdf_filename'] = pdf_filename
        # Download PDF
        safe_filename = pdf_filename.replace('/', '_').replace(' ', '_')
        if not safe_filename.lower().endswith('.pdf'):
            safe_filename += '.pdf'
        item['pdf_path'] = download_pdf(pdf_url, safe_filename)
    else:
        item['pdf_url'] = None
        item['pdf_filename'] = None
        item['pdf_path'] = None
    # Validation
    required = ['case_number', 'decision_date', 'court_name']
    errors = [f for f in required if not item.get(f)]
    item['is_complete'] = len(errors) == 0
    item['validation_errors'] = errors
    return item

def main():
    page_url = BASE_URL
    total_cases = 0
    pdfs_downloaded = 0
    incomplete_cases = 0
    errors = []
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        for page in range(1, MAX_PAGES + 1):
            print(f"Processing page {page}: {page_url}")
            soup = get_soup(page_url)
            for case_div in soup.select('div.spost.clearfix'):
                link = case_div.select_one('strong > a')
                if link and link.get('href'):
                    case_url = link['href']
                    print(f"  - Scraping case: {case_url}")
                    try:
                        item = parse_case(case_url)
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')
                        total_cases += 1
                        if item.get('pdf_path'):
                            pdfs_downloaded += 1
                        if not item['is_complete']:
                            incomplete_cases += 1
                    except Exception as e:
                        print(f"    Error scraping {case_url}: {e}")
                        errors.append({'url': case_url, 'error': str(e)})
                    time.sleep(1)  # politeness
            # Find next page
            next_page = None
            for a in soup.select('div.pagging.text-center ul.pagination a.page-link'):
                if a.get_text(strip=True) == 'Next':
                    next_page = a.get('href')
                    break
            if not next_page:
                break
            page_url = next_page
            time.sleep(2)  # politeness
    print(f"""
Summary:
  - Cases scraped: {total_cases}
  - PDFs downloaded: {pdfs_downloaded}
  - Incomplete cases: {incomplete_cases}
  - Errors: {len(errors)}
""")
    if errors:
        print("Some errors occurred:")
        for err in errors:
            print(f"  {err['url']}: {err['error']}")

if __name__ == "__main__":
    main()
