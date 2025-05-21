import asyncio
import httpx
from lxml import html
import aiofiles
import os
import json
from typing import List, Dict, Optional, Set
import random
import traceback
import logging
from tqdm import tqdm
import re
from dataclasses import dataclass
import time
from collections import deque
import gc
import psutil

# Configuration Constants
BASE_URL = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/perceraian.html"
OUTPUT_JSON = "test_output.jsonl"  # Changed for testing
PDF_DIR = "test_pdfs"  # Changed for testing
MAX_PAGES = 2  # Reduced from 499 to 2 for testing
INITIAL_CONCURRENT_WORKERS = 2  # Reduced from 12 to 2
PAGE_BATCH_SIZE = 2  # Reduced from 25 to 2
REQUEST_TIMEOUT = 60.0
POLITENESS_DELAY = (1.5, 3.0)
BATCH_SIZE = 5  # Reduced from 20 to 5
CHECKPOINT_FILE = 'test_checkpoint.txt'  # Changed for testing
MAX_RETRIES = 3
RETRY_DELAYS = [3, 7, 15]
MAX_CONCURRENT_PDF_DOWNLOADS = 2  # Reduced from 5 to 2
MAX_PENDING_TASKS = 10  # Reduced from 50 to 10
MEMORY_THRESHOLD = 80

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
    pages_processed: int = 0
    errors: List[Dict] = None
    start_time: float = None
    response_times: deque = None
    current_concurrency: int = None
    
    def __post_init__(self):
        self.errors = []
        self.start_time = time.time()
        self.response_times = deque(maxlen=20)
        self.current_concurrency = INITIAL_CONCURRENT_WORKERS
    
    def __str__(self):
        elapsed = time.time() - self.start_time
        pages_per_hour = (self.pages_processed / max(elapsed, 1)) * 3600
        avg_response = sum(self.response_times)/len(self.response_times) if self.response_times else 0
        return (f"Pages: {self.pages_processed} ({pages_per_hour:.1f}/hr) | "
                f"Cases: {self.total_cases} | PDFs: {self.pdfs_downloaded} | "
                f"Incomplete: {self.incomplete_cases} | Errors: {len(self.errors)} | "
                f"Concurrency: {self.current_concurrency} | Avg Resp: {avg_response:.2f}s")

class TaskQueue:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.queue = asyncio.Queue(maxsize=maxsize)
    
    async def put(self, task):
        await self.queue.put(task)
    
    async def get(self):
        return await self.queue.get()
    
    def task_done(self):
        self.queue.task_done()
    
    async def join(self):
        await self.queue.join()
    
    def qsize(self):
        return self.queue.qsize()

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

def extract_pdf_info(tree) -> tuple:
    """Extract PDF URL and filename from case page using lxml"""
    for card in tree.xpath('//div[contains(@class, "card") and contains(@class, "bg-success")]'):
        for span in card.xpath('.//span'):
            icon = span.xpath('.//i[contains(@class, "icon-files")]')
            if icon and 'Download PDF' in span.text_content().strip():
                for a in card.xpath('.//a'):
                    link_text = a.text_content().strip()
                    if link_text.endswith('.pdf'):
                        return a.get('href'), link_text
    return None, None

def extract_case_links(tree) -> List[str]:
    """Extract case detail page links from listing page using lxml"""
    return list(dict.fromkeys(
        link.get('href') 
        for link in tree.xpath('//div[@class="spost clearfix"]//strong/a[@href]')
    ))

def find_next_page(tree) -> Optional[str]:
    """Find URL for next page in pagination using lxml"""
    for a in tree.xpath('//div[contains(@class, "pagging")]//a[contains(@class, "page-link")]'):
        if a.text_content().strip() == 'Next':
            return a.get('href')
    return None

def extract_page_number(url: str) -> int:
    """Extract page number from URL"""
    match = re.search(r'/page/(\d+)\.html', url)
    return int(match.group(1)) if match else 1

async def get_tree(client: httpx.AsyncClient, url: str, stats: ScraperStats) -> html.HtmlElement:
    """Fetch and parse HTML page with error handling using lxml"""
    start_time = time.time()
    logging.info(f"Fetching page: {url}")
    try:
        await asyncio.sleep(random.uniform(*POLITENESS_DELAY))
        resp = await client.get(
            url,
            headers=get_random_headers(url),
            timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        
        response_time = time.time() - start_time
        stats.response_times.append(response_time)
        
        return html.fromstring(resp.text)
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

async def parse_case(client: httpx.AsyncClient, case_url: str, stats: ScraperStats) -> Dict:
    """Parse case details from case page using lxml"""
    item = {'url': case_url, 'is_complete': False}
    
    try:
        logging.info(f"Parsing case: {case_url}")
        tree = await get_tree(client, case_url, stats)
        
        # Extract table data using XPath
        table = tree.xpath('//div[contains(@class, "tab-content")]//table[contains(@class, "table")]')
        if table:
            table = table[0]
            for row in table.xpath('.//tr'):
                tds = row.xpath('.//td')
                if len(tds) < 2:
                    continue
                label = tds[0].text_content().strip()
                value = tds[1].text_content().strip()
                
                if label == 'Nomor':
                    item['case_number'] = value
                elif label == 'Tingkat Proses':
                    item['process_level'] = value
                elif label == 'Klasifikasi':
                    item['classification'] = ' '.join(
                        a.text_content().strip() 
                        for a in tds[1].xpath('.//a')
                    )
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
                    item['catatan_amar'] = tds[1].text_content().strip()  
                elif label == 'Tanggal Musyawarah':
                    item['deliberation_date'] = parse_date(value)
                elif label == 'Tanggal Dibacakan':
                    item['decision_date'] = parse_date(value)
                elif label == 'Kaidah':
                    item['kaidah'] = value
                elif label == 'Abstrak':
                    item['abstrak'] = value
        
        # Extract statistics using XPath
        stats_card = tree.xpath('//div[@id="collapseTen"]//div[contains(@class, "card-body")]')
        if stats_card:
            stats_card = stats_card[0]
            view_div = stats_card.xpath('.//div[@title="Jumlah view"]')
            download_div = stats_card.xpath('.//div[@title="Jumlah download"]')
            if view_div:
                item['jumlah_view'] = view_div[0].text_content().strip().replace('\xa0', ' ')
            if download_div:
                item['jumlah_download'] = download_div[0].text_content().strip().replace('\xa0', ' ')
        
        # Handle PDF download
        pdf_url, pdf_filename = extract_pdf_info(tree)
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
        
    except Exception as e:
        item['parse_error'] = str(e)
        item['traceback'] = traceback.format_exc()
        logging.error(f"Error parsing case {case_url}: {e}")
    
    return item

async def write_jsonl_batch(items: List[Dict], lock: asyncio.Lock):
    """Write batch of items to JSONL file with locking"""
    if not items:
        return
    
    # Create a copy of the batch to avoid holding the lock during IO
    batch_copy = items.copy()
    
    async with lock:
        try:
            async with aiofiles.open(OUTPUT_JSON, 'a', encoding='utf-8') as f:
                for item in batch_copy:
                    await f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as e:
            logging.error(f"Error writing batch: {e}")

async def worker(case_url: str, client: httpx.AsyncClient, lock: asyncio.Lock, 
                stats: ScraperStats, batch: List[Dict], batch_lock: asyncio.Lock,
                pdf_semaphore: asyncio.Semaphore, existing_case_numbers: Set[str]):
    """Worker function to process a single case with retries"""
    for attempt in range(MAX_RETRIES):
        try:
            await asyncio.sleep(attempt * 2)  # Progressive delay
            
            # First check if PDF exists to avoid full parse
            tree = await get_tree(client, case_url, stats)
            pdf_url, pdf_filename = extract_pdf_info(tree)
            
            if pdf_url and pdf_filename:
                safe_filename = sanitize_filename(pdf_filename)
                pdf_path = os.path.join(PDF_DIR, safe_filename)
                if os.path.exists(pdf_path):
                    logging.info(f"Skipping case with existing PDF: {case_url}")
                    async with batch_lock:
                        stats.total_cases += 1
                        stats.pdfs_downloaded += 1
                    return

            # Check for duplicate case number
            case_number = None
            table = tree.xpath('//div[contains(@class, "tab-content")]//table')
            if table:
                table = table[0]
                for row in table.xpath('.//tr'):
                    tds = row.xpath('.//td')
                    if len(tds) >= 2 and tds[0].text_content().strip() == 'Nomor':
                        case_number = tds[1].text_content().strip()
                        break
            
            if case_number and case_number in existing_case_numbers:
                logging.info(f"Skipping duplicate case: {case_number}")
                async with batch_lock:
                    stats.total_cases += 1
                return

            # Full case processing
            async with pdf_semaphore:
                item = await parse_case(client, case_url, stats)
            
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
                wait = RETRY_DELAYS[attempt] + random.uniform(1, 3)
                logging.warning(f"Attempt {attempt+1} failed, waiting {wait:.1f}s")
                await asyncio.sleep(wait)
                continue
            async with batch_lock:
                stats.errors.append({
                    'url': case_url,
                    'error': str(e),
                    'type': type(e).__name__,
                    'attempt': attempt
                })
            break
        except Exception as e:
            async with batch_lock:
                stats.errors.append({
                    'url': case_url,
                    'error': str(e),
                    'type': type(e).__name__,
                    'trace': traceback.format_exc(),
                    'attempt': attempt
                })
            break

async def load_existing_case_numbers() -> Set[str]:
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

def check_memory_usage():
    """Check current memory usage and adjust concurrency if needed"""
    mem = psutil.virtual_memory()
    return mem.percent

async def adjust_concurrency(stats: ScraperStats):
    """Adjust concurrency based on performance and memory"""
    while True:
        await asyncio.sleep(60)  # Check every minute
        
        # Check memory usage
        mem_usage = check_memory_usage()
        if mem_usage > MEMORY_THRESHOLD:
            new_concurrency = max(1, stats.current_concurrency - 2)
            logging.warning(f"High memory usage ({mem_usage}%), reducing concurrency to {new_concurrency}")
            stats.current_concurrency = new_concurrency
            continue
        
        # Adjust based on response times if we have enough data
        if len(stats.response_times) >= 10:
            avg_response = sum(stats.response_times) / len(stats.response_times)
            
            if avg_response > 5.0 and stats.current_concurrency > 1:
                # Response times are slow, reduce concurrency
                new_concurrency = max(1, stats.current_concurrency - 1)
                logging.info(f"High response time ({avg_response:.2f}s), reducing concurrency to {new_concurrency}")
                stats.current_concurrency = new_concurrency
            elif avg_response < 2.0 and stats.current_concurrency < INITIAL_CONCURRENT_WORKERS * 2:
                # Response times are fast, increase concurrency
                new_concurrency = min(INITIAL_CONCURRENT_WORKERS * 2, stats.current_concurrency + 1)
                logging.info(f"Low response time ({avg_response:.2f}s), increasing concurrency to {new_concurrency}")
                stats.current_concurrency = new_concurrency

async def process_page_batch(client: httpx.AsyncClient, page_batch: List[str], 
                           stats: ScraperStats, existing_case_numbers: Set[str]):
    """Process a batch of pages with enhanced error handling"""
    lock = asyncio.Lock()
    batch_lock = asyncio.Lock()
    pdf_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PDF_DOWNLOADS)
    batch = []
    retry_queue = deque()
    task_queue = TaskQueue(MAX_PENDING_TASKS)
    
    # Start concurrency adjustment task
    adjust_task = asyncio.create_task(adjust_concurrency(stats))
    
    try:
        # First pass - process all cases
        for page_url in page_batch:
            try:
                tree = await get_tree(client, page_url, stats)
                case_urls = extract_case_links(tree)
                
                # Filter out cases we've already processed
                filtered_case_urls = []
                for url in case_urls:
                    if url in existing_case_numbers:
                        continue
                    filtered_case_urls.append(url)
                
                logging.info(f"Found {len(filtered_case_urls)} new cases on page {page_url}")
                
                # Create tasks with controlled concurrency
                # Create a task processor
                async def process_tasks():
                    while True:
                        task = await task_queue.get()
                        try:
                            await task
                        except Exception as e:
                            logging.error(f"Task failed: {e}")
                        finally:
                            task_queue.task_done()
                
                # Start the processor before queuing tasks
                processor = asyncio.create_task(process_tasks())
                
                # Then queue the tasks
                for url in filtered_case_urls:
                    task = asyncio.create_task(worker(url, client, lock, stats, batch, batch_lock, pdf_semaphore, existing_case_numbers))
                    await task_queue.put(task)
                
                # Wait for all tasks to complete
                await task_queue.join()
                processor.cancel()
                
                stats.pages_processed += 1
                logging.info(f"Progress: {stats}")
                
            except Exception as e:
                logging.error(f"Failed to process page {page_url}: {e}")
                retry_queue.append(page_url)
        
        # Retry failed pages
        if retry_queue:
            logging.info(f"Retrying {len(retry_queue)} failed pages...")
            await process_page_batch(client, list(retry_queue), stats, existing_case_numbers)
        
        # Write any remaining batch items
        if batch:
            await write_jsonl_batch(batch, lock)
        
    finally:
        adjust_task.cancel()
        try:
            await adjust_task
        except asyncio.CancelledError:
            pass
        
        # Clear memory
        gc.collect()

async def main():
    """Main scraping function"""
    logging.info("Starting optimized scraper for 499 pages")
    stats = ScraperStats()

    # Configure HTTP client
    transport = httpx.AsyncHTTPTransport()
    limits = httpx.Limits(
        max_connections=INITIAL_CONCURRENT_WORKERS * 2,
        max_keepalive_connections=INITIAL_CONCURRENT_WORKERS
    )

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        limits=limits,
        transport=transport
    ) as client:
        # Load existing case numbers once
        existing_case_numbers = await load_existing_case_numbers()
        
        # Determine starting page
        current_page = await get_start_page()

        # Process in page batches
        while current_page <= MAX_PAGES:
            page_batch = range(current_page, min(current_page + PAGE_BATCH_SIZE, MAX_PAGES + 1))
            page_urls = [
                BASE_URL if n == 1 else 
                f"https://putusan3.mahkamahagung.go.id/direktori/index/kategori/perceraian/page/{n}.html"
                for n in page_batch
            ]

            await process_page_batch(client, page_urls, stats, existing_case_numbers)

            # Update checkpoint after successful batch completion
            await update_checkpoint(current_page + len(page_batch) - 1)

            # Move to next batch
            current_page += PAGE_BATCH_SIZE

    # Final stats
    logging.info(f"Scraping complete. Final stats: {stats}")
    if stats.errors:
        logging.error(f"Encountered {len(stats.errors)} errors")
        for error in stats.errors[:5]:  # Show first 5 errors
            logging.error(f"Error on {error['url']}: {error['error']}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Scraper stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        traceback.print_exc()