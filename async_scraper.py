import asyncio
import httpx
from bs4 import BeautifulSoup
import aiofiles
import os
import json
from typing import List
import random
import traceback
import logging
from tqdm.asyncio import tqdm

BASE_URL = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/perceraian.html"
OUTPUT_JSON = "output_async.jsonl"
PDF_DIR = "pdfs_async"
MAX_PAGES = 1  # For debugging, only process 1 page
CONCURRENT_WORKERS = 8  # For debugging, only 1 worker
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 MyScraper/1.0'

os.makedirs(PDF_DIR, exist_ok=True)

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

# Enhanced browser-like headers
HEADERS = {
    'User-Agent': USER_AGENT,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': BASE_URL,
    'Connection': 'keep-alive',
    'DNT': '1',
    'Upgrade-Insecure-Requests': '1',
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

async def get_soup(client: httpx.AsyncClient, url: str) -> BeautifulSoup:
    logging.info(f"Fetching page: {url}")
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        raise

async def download_pdf(client: httpx.AsyncClient, pdf_url: str, filename: str) -> str:
    path = os.path.join(PDF_DIR, filename)
    if os.path.exists(path):
        logging.info(f"PDF already exists: {filename}")
        return path
    logging.info(f"Downloading PDF: {pdf_url} -> {filename}")
    try:
        async with client.stream("GET", pdf_url) as r:
            r.raise_for_status()
            async with aiofiles.open(path, "wb") as f:
                async for chunk in r.aiter_bytes():
                    await f.write(chunk)
        return path
    except Exception as e:
        logging.error(f"Error downloading PDF {pdf_url}: {e}")
        raise

def extract_case_links(soup: BeautifulSoup) -> List[str]:
    links = []
    for case_div in soup.select('div.spost.clearfix'):
        link = case_div.select_one('strong > a')
        if link and link.get('href'):
            links.append(link['href'])
    return links

def find_next_page(soup: BeautifulSoup) -> str:
    for a in soup.select('div.pagging.text-center ul.pagination a.page-link'):
        if a.get_text(strip=True) == 'Next':
            return a.get('href')
    return None

async def parse_case(client: httpx.AsyncClient, case_url: str) -> dict:
    logging.info(f"Parsing case: {case_url}")
    soup = await get_soup(client, case_url)
    item = {}
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
    # Extract statistics: Jumlah view and Jumlah download
    stats_card = soup.select_one('div#collapseTen .card-body .row')
    if stats_card:
        view_div = stats_card.select_one("div[title='Jumlah view']")
        download_div = stats_card.select_one("div[title='Jumlah download']")
        if view_div:
            item['jumlah_view'] = view_div.get_text(strip=True).replace('\xa0', ' ')
        if download_div:
            item['jumlah_download'] = download_div.get_text(strip=True).replace('\xa0', ' ')
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
        safe_filename = pdf_filename.replace('/', '_').replace(' ', '_')
        if not safe_filename.lower().endswith('.pdf'):
            safe_filename += '.pdf'
        try:
            item['pdf_path'] = await download_pdf(client, pdf_url, safe_filename)
        except Exception as e:
            item['pdf_path'] = None
            item['pdf_download_error'] = str(e)
    else:
        item['pdf_url'] = None
        item['pdf_filename'] = None
        item['pdf_path'] = None
    required = ['case_number', 'decision_date', 'court_name']
    errors = [f for f in required if not item.get(f)]
    item['is_complete'] = len(errors) == 0
    item['validation_errors'] = errors
    return item

async def write_jsonl(item, lock):
    async with lock:
        async with aiofiles.open(OUTPUT_JSON, 'a', encoding='utf-8') as f:
            await f.write(json.dumps(item, ensure_ascii=False) + '\n')

async def worker(case_url, client, lock, stats, max_retries=5):
    for attempt in range(max_retries):
        try:
            logging.info(f"[Worker] Attempt {attempt+1} for {case_url}")
            item = await parse_case(client, case_url)
            await write_jsonl(item, lock)
            stats['total_cases'] += 1
            if item.get('pdf_path'):
                stats['pdfs_downloaded'] += 1
            if not item['is_complete']:
                stats['incomplete_cases'] += 1
            break  # Success, exit retry loop
        except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as e:
            code = getattr(getattr(e, 'response', None), 'status_code', None)
            logging.error(f"HTTP/network error on {case_url}: {type(e).__name__}: {e} (status: {code})")
            if (code and 500 <= code < 600) or isinstance(e, httpx.TimeoutException):
                if attempt < max_retries - 1:
                    wait = 2 ** attempt + random.uniform(1, 2)
                    logging.info(f"Retrying in {wait:.2f}s (attempt {attempt+1})")
                    await asyncio.sleep(wait)
                    continue
            stats['errors'].append({'url': case_url, 'error': str(e), 'type': type(e).__name__, 'trace': traceback.format_exc()})
            break
        except Exception as e:
            logging.error(f"Error scraping {case_url}: {e}")
            stats['errors'].append({'url': case_url, 'error': str(e), 'type': type(e).__name__, 'trace': traceback.format_exc()})
            break
        await asyncio.sleep(random.uniform(2.0, 3.0))
    else:
        logging.error(f"Failed after {max_retries} attempts: {case_url}")

async def main():
    logging.info("Scraper started.")
    page_url = BASE_URL
    all_case_urls = []
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as client:
        for page in range(1, MAX_PAGES + 1):
            logging.info(f"Processing page {page}: {page_url}")
            try:
                soup = await get_soup(client, page_url)
            except Exception as e:
                logging.error(f"Failed to fetch page {page_url}: {e}")
                break
            case_urls = extract_case_links(soup)
            logging.info(f"Found {len(case_urls)} case links on page {page}")
            all_case_urls.extend(case_urls)
            next_page = find_next_page(soup)
            if not next_page:
                logging.info("No next page found.")
                break
            page_url = next_page
    # Concurrency
    lock = asyncio.Lock()
    stats = {'total_cases': 0, 'pdfs_downloaded': 0, 'incomplete_cases': 0, 'errors': []}
    sem = asyncio.Semaphore(CONCURRENT_WORKERS)
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as client:
        async def sem_worker(url):
            async with sem:
                await worker(url, client, lock, stats)
        # Progress bar for async tasks
        tasks = [sem_worker(url) for url in all_case_urls]
        for f in tqdm.as_completed(tasks, total=len(tasks), desc='Cases processed'):
            await f
    logging.info(f"Summary: Cases scraped: {stats['total_cases']}, PDFs downloaded: {stats['pdfs_downloaded']}, Incomplete cases: {stats['incomplete_cases']}, Errors: {len(stats['errors'])}")
    if stats['errors']:
        logging.error("Some errors occurred:")
        for err in stats['errors']:
            logging.error(f"  {err['url']}: {err['error']}")
    logging.info("Scraper finished.")

if __name__ == "__main__":
    asyncio.run(main()) 