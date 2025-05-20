import asyncio
import httpx
from bs4 import BeautifulSoup
import aiofiles
import os
import json
from typing import List, Dict, Optional
import random
import traceback
import logging
from tqdm import tqdm
import re
from dataclasses import dataclass

# Configuration Constants
BASE_URL = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/perceraian.html"
OUTPUT_JSON = "output_async.jsonl"
PDF_DIR = "pdfs_async"
MAX_PAGES = 2
CONCURRENT_WORKERS = 3
REQUEST_TIMEOUT = 45.0
POLITENESS_DELAY = (1.5, 3.0)
BATCH_SIZE = 10
CHECKPOINT_FILE = 'scraper_checkpoint.txt'
MAX_RETRIES = 3

# Enhanced User Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Mobile/15E148 Safari/604.1"
]

# Date parsing
MONTHS = {
    'Januari': '01', 'Februari': '02', 'Maret': '03', 'April': '04',
    'Mei': '05', 'Juni': '06', 'Juli': '07', 'Agustus': '08',
    'September': '09', 'Oktober': '10', 'November': '11', 'Desember': '12'
}

@dataclass
class ScraperStats:
    total_cases: int = 0
    pdfs_downloaded: int = 0
    incomplete_cases: int = 0
    errors: List[Dict] = None
    
    def __post_init__(self):
        self.errors = []

    def __str__(self):
        return (f"Cases: {self.total_cases}, PDFs: {self.pdfs_downloaded}, "
                f"Incomplete: {self.incomplete_cases}, Errors: {len(self.errors)}")

# Setup directories
os.makedirs(PDF_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def parse_date(date_str: str) -> Optional[str]:
    """Parse Indonesian date string into YYYY-MM-DD format"""
    if not date_str:
        return None
    parts = date_str.strip().split()
    if len(parts) == 3:
        day, month, year = parts
        month = MONTHS.get(month, '01')
        return f"{year}-{month}-{int(day):02d}"
    return date_str

def get_random_headers(referer: Optional[str] = None) -> Dict[str, str]:
    """Generate random headers for requests"""
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': referer or BASE_URL,
        'Connection': 'keep-alive',
        'DNT': '1',
        'Cache-Control': 'max-age=0'
    }
    if random.random() > 0.3:
        headers['Upgrade-Insecure-Requests'] = '1'
    return headers

def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing/replacing problematic characters"""
    safe = re.sub(r'[\\/*?:"<>|]', "_", filename)
    safe = safe.replace(' ', '_')
    if not safe.lower().endswith('.pdf'):
        safe += '.pdf'
    return safe

def extract_pdf_info(soup: BeautifulSoup) -> tuple:
    """Extract PDF URL and filename from case page"""
    for card in soup.select('div.card.bg-success.mb-3'):
        for span in card.select('span'):
            icon = span.find('i', class_='icon-files')
            if icon and 'Download PDF' in span.get_text(strip=True):
                for a in card.select('a'):
                    link_text = a.get_text(strip=True)
                    if link_text.endswith('.pdf'):
                        return a.get('href'), link_text
    return None, None

def extract_case_links(soup: BeautifulSoup) -> List[str]:
    """Extract case detail page links from listing page"""
    return list(dict.fromkeys(
        link['href'] for link in soup.select('div.spost.clearfix strong > a[href]')
    ))

def find_next_page(soup: BeautifulSoup) -> Optional[str]:
    """Find URL for next page in pagination"""
    for a in soup.select('div.pagging.text-center ul.pagination a.page-link'):
        if a.get_text(strip=True) == 'Next':
            return a.get('href')
    return None

def extract_page_number(url: str) -> int:
    """Extract page number from URL"""
    match = re.search(r'/page/(\d+)\.html', url)
    return int(match.group(1)) if match else 1

async def get_soup(client: httpx.AsyncClient, url: str) -> BeautifulSoup:
    """Fetch and parse HTML page with error handling"""
    logging.info(f"Fetching page: {url}")
    try:
        await asyncio.sleep(random.uniform(*POLITENESS_DELAY))
        resp = await client.get(
            url,
            headers=get_random_headers(url),
            timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        
        if "cloudflare" in resp.headers.get('server', '').lower():
            raise httpx.HTTPStatusError(
                "Cloudflare protection detected",
                request=resp.request,
                response=resp
            )
            
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        logging.error(f"Error fetching {url}: {str(e)[:200]}")
        raise

async def download_pdf(client: httpx.AsyncClient, pdf_url: str, filename: str) -> str:
    """Download PDF file with error handling"""
    path = os.path.join(PDF_DIR, filename)
    if os.path.exists(path):
        logging.info(f"PDF exists: {filename}")
        return path
        
    logging.info(f"Downloading PDF: {pdf_url}")
    try:
        async with client.stream(
            "GET",
            pdf_url,
            headers=get_random_headers(pdf_url),
            timeout=REQUEST_TIMEOUT
        ) as r:
            r.raise_for_status()
            async with aiofiles.open(path, "wb") as f:
                async for chunk in r.aiter_bytes():
                    await f.write(chunk)
        return path
    except Exception as e:
        logging.error(f"Error downloading PDF {pdf_url}: {str(e)[:200]}")
        raise

async def parse_case(client: httpx.AsyncClient, case_url: str) -> Dict:
    """Parse case details from case page"""
    logging.info(f"Parsing case: {case_url}")
    soup = await get_soup(client, case_url)
    item = {'url': case_url}
    
    # Extract table data
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
                item['register_date'] = parse_date(value)
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
                item['deliberation_date'] = parse_date(value)
            elif label == 'Tanggal Dibacakan':
                item['decision_date'] = parse_date(value)
            elif label == 'Kaidah':
                item['kaidah'] = value
            elif label == 'Abstrak':
                item['abstrak'] = value
    
    # Extract statistics
    stats_card = soup.select_one('div#collapseTen .card-body .row')
    if stats_card:
        view_div = stats_card.select_one("div[title='Jumlah view']")
        download_div = stats_card.select_one("div[title='Jumlah download']")
        if view_div:
            item['jumlah_view'] = view_div.get_text(strip=True).replace('\xa0', ' ')
        if download_div:
            item['jumlah_download'] = download_div.get_text(strip=True).replace('\xa0', ' ')
    
    # Handle PDF download
    pdf_url, pdf_filename = extract_pdf_info(soup)
    if pdf_url and pdf_filename:
        item['pdf_url'] = pdf_url
        item['pdf_filename'] = pdf_filename
        safe_filename = sanitize_filename(pdf_filename)
        try:
            item['pdf_path'] = await download_pdf(client, pdf_url, safe_filename)
        except Exception as e:
            item['pdf_path'] = None
            item['pdf_download_error'] = str(e)
    else:
        item['pdf_url'] = None
        item['pdf_filename'] = None
        item['pdf_path'] = None
    
    # Validate required fields
    required = ['case_number', 'decision_date', 'court_name']
    errors = [f for f in required if not item.get(f)]
    item['is_complete'] = len(errors) == 0
    item['validation_errors'] = errors
    
    return item

async def write_jsonl_batch(items: List[Dict], lock: asyncio.Lock):
    """Write batch of items to JSONL file with locking"""
    if not items:
        return
    async with lock:
        async with aiofiles.open(OUTPUT_JSON, 'a', encoding='utf-8') as f:
            for item in items:
                await f.write(json.dumps(item, ensure_ascii=False) + '\n')

async def worker(case_url: str, client: httpx.AsyncClient, lock: asyncio.Lock, 
                stats: ScraperStats, batch: List[Dict], batch_lock: asyncio.Lock):
    """Worker function to process a single case with retries"""
    for attempt in range(MAX_RETRIES):
        try:
            await asyncio.sleep(attempt * 2)  # Progressive delay
            
            # First check if PDF exists to avoid full parse
            soup = await get_soup(client, case_url)
            pdf_url, pdf_filename = extract_pdf_info(soup)
            
            if pdf_url and pdf_filename:
                safe_filename = sanitize_filename(pdf_filename)
                pdf_path = os.path.join(PDF_DIR, safe_filename)
                if os.path.exists(pdf_path):
                    logging.info(f"Skipping case with existing PDF: {case_url}")
                    stats.total_cases += 1
                    stats.pdfs_downloaded += 1
                    return

            # Full case processing
            item = await parse_case(client, case_url)
            
            async with batch_lock:
                batch.append(item)
                if len(batch) >= BATCH_SIZE:
                    await write_jsonl_batch(batch, lock)
                    batch.clear()
                    
            stats.total_cases += 1
            if item.get('pdf_path'):
                stats.pdfs_downloaded += 1
            if not item.get('is_complete', True):
                stats.incomplete_cases += 1
                
            break  # Success
            
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            if code == 503 or isinstance(e, httpx.TimeoutException):
                wait = (2 ** attempt) + random.uniform(1, 3)
                logging.warning(f"Attempt {attempt+1} failed, waiting {wait:.1f}s")
                await asyncio.sleep(wait)
                continue
            stats.errors.append({
                'url': case_url,
                'error': str(e),
                'type': type(e).__name__,
                'attempt': attempt
            })
            break
        except Exception as e:
            stats.errors.append({
                'url': case_url,
                'error': str(e),
                'type': type(e).__name__,
                'trace': traceback.format_exc(),
                'attempt': attempt
            })
            break

async def load_existing_case_numbers() -> set:
    """Load existing case numbers from output file to avoid duplicates"""
    existing = set()
    if os.path.exists(OUTPUT_JSON):
        try:
            async with aiofiles.open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                async for line in f:
                    try:
                        obj = json.loads(line)
                        if 'case_number' in obj:
                            existing.add(obj['case_number'])
                    except json.JSONDecodeError:
                        continue
            logging.info(f"Loaded {len(existing)} existing case numbers")
        except Exception as e:
            logging.error(f"Error reading {OUTPUT_JSON}: {e}")
    return existing

async def get_start_page() -> int:
    """Determine starting page from checkpoint file"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                last_page = int(f.read().strip())
                logging.info(f"Resuming from page {last_page + 1}")
                return last_page + 1
        except Exception as e:
            logging.error(f"Error reading checkpoint: {e}")
    return 1

async def update_checkpoint(page_number: int):
    """Update checkpoint file with last completed page"""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            f.write(str(page_number))
    except Exception as e:
        logging.error(f"Error writing checkpoint: {e}")

async def main():
    """Main scraping function"""
    logging.info("Starting refined scraper")
    stats = ScraperStats()
    lock = asyncio.Lock()
    batch_lock = asyncio.Lock()
    batch = []
    
    # Load existing data for deduplication
    existing_case_numbers = await load_existing_case_numbers()
    
    # Configure HTTP client
    transport = httpx.AsyncHTTPTransport(retries=2)
    limits = httpx.Limits(
        max_connections=CONCURRENT_WORKERS * 2,
        max_keepalive_connections=CONCURRENT_WORKERS
    )
    
    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        limits=limits,
        transport=transport
    ) as client:
        # Determine starting page
        current_page = await get_start_page()
        page_url = BASE_URL if current_page == 1 else \
            f"https://putusan3.mahkamahagung.go.id/direktori/index/kategori/perceraian/page/{current_page}.html"
        
        while current_page <= MAX_PAGES:
            try:
                logging.info(f"Processing page {current_page}: {page_url}")
                soup = await get_soup(client, page_url)
                case_urls = extract_case_links(soup)
                
                # Filter out cases we've already processed
                filtered_case_urls = []
                for url in case_urls:
                    try:
                        # Quick check for case number without full parse
                        case_soup = await get_soup(client, url)
                        case_number = None
                        table = case_soup.select_one('div.tab-content.ui-tabs-panel table.table')
                        if table:
                            for row in table.select('tr'):
                                tds = row.select('td')
                                if len(tds) >= 2 and tds[0].get_text(strip=True) == 'Nomor':
                                    case_number = tds[1].get_text(strip=True)
                                    break
                        if case_number and case_number in existing_case_numbers:
                            logging.info(f"Skipping duplicate case: {case_number}")
                            continue
                        filtered_case_urls.append(url)
                    except Exception as e:
                        logging.error(f"Error checking case {url}: {e}")
                        continue
                
                logging.info(f"Found {len(filtered_case_urls)} new cases on page {current_page}")
                
                # Process cases with limited concurrency
                semaphore = asyncio.Semaphore(CONCURRENT_WORKERS)
                
                async def process_case(url):
                    async with semaphore:
                        await worker(url, client, lock, stats, batch, batch_lock)
                
                # Create all tasks first
                tasks = [process_case(url) for url in filtered_case_urls]
                
                # Process with progress bar - FIXED tqdm usage
                for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Page {current_page}"):
                    await task
                
                # Update checkpoint after successful page completion
                await update_checkpoint(current_page)
                
                # Move to next page
                next_page = find_next_page(soup)
                if not next_page:
                    logging.info("No next page found")
                    break
                    
                page_url = next_page
                current_page += 1
                
            except Exception as e:
                logging.error(f"Failed to process page {current_page}: {e}")
                break
    
    # Write any remaining batch items
    if batch:
        await write_jsonl_batch(batch, lock)
    
    # Final stats
    logging.info(f"Scraping complete. Stats: {stats}")
    if stats.errors:
        logging.error(f"Encountered {len(stats.errors)} errors")
        for error in stats.errors[:5]:  # Show first 5 errors
            logging.error(f"Error on {error['url']}: {error['error']}")

if __name__ == "__main__":
    asyncio.run(main())