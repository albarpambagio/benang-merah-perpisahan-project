import asyncio
import httpx
from lxml import html
import aiofiles
import os
import json
from typing import List, Dict, Optional, Set, Deque
import random
import traceback
import logging
from tqdm import tqdm
import re
from dataclasses import dataclass, field
import time
from collections import deque
import gc
import psutil
from datetime import datetime
import hashlib

# Configuration Constants
BASE_URL = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/perceraian.html"
OUTPUT_JSON = "test_output.jsonl"
PDF_DIR = "test_pdfs"
MAX_PAGES = 2  # Test with small number first
INITIAL_CONCURRENT_WORKERS = 3  # Lower initial concurrency
PAGE_BATCH_SIZE = 2
# Update timeout configuration at the top of the file
CONNECT_TIMEOUT = 10.0  # Time to establish connection
READ_TIMEOUT = 45.0    # Time to wait for server response (HTML)
WRITE_TIMEOUT = 20.0   # Time to send request
POOL_TIMEOUT = 10.0    # Time to wait for connection from pool
PDF_READ_TIMEOUT = 90.0  # Separate, longer timeout for PDF downloads

# Replace simple REQUEST_TIMEOUT with detailed configuration
REQUEST_TIMEOUT = httpx.Timeout(
    connect=CONNECT_TIMEOUT,
    read=READ_TIMEOUT,
    write=WRITE_TIMEOUT,
    pool=POOL_TIMEOUT
)
PDF_REQUEST_TIMEOUT = httpx.Timeout(
    connect=CONNECT_TIMEOUT,
    read=PDF_READ_TIMEOUT,
    write=WRITE_TIMEOUT,
    pool=POOL_TIMEOUT
)
POLITENESS_DELAY = (1.0, 2.0)  # More aggressive but still polite
BATCH_SIZE = 10
CHECKPOINT_FILE = 'test_checkpoint.txt'
MAX_RETRIES = 5  # Increased from 3
MAX_PDF_RETRIES = 5  # Specialized retry for PDFs
RETRY_DELAYS = [2, 5, 10, 20, 30]  # More retry stages
MAX_CONCURRENT_PDF_DOWNLOADS = 3
MAX_PENDING_TASKS = 15  # Increased from 10
MEMORY_THRESHOLD = 85  # Slightly higher threshold
MIN_CONCURRENCY = 2  # Never drop below this
MAX_CONCURRENCY = 10  # Absolute maximum
RESPONSE_TIME_TARGET = 4.0  # Target response time in seconds
CONCURRENCY_ADJUST_INTERVAL = 60  # Gradual concurrency adjustment (seconds)
CIRCUIT_BREAKER_WINDOW = 30  # Number of requests to consider for circuit breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 0.5  # 50% failure rate triggers breaker
CIRCUIT_BREAKER_COOLDOWN = 60  # seconds to pause when breaker is open

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
    errors: List[Dict] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    response_times: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    current_concurrency: int = INITIAL_CONCURRENT_WORKERS
    active_tasks: int = 0
    # Enhanced metrics
    html_success: int = 0
    html_fail: int = 0
    pdf_success: int = 0
    pdf_fail: int = 0
    recent_statuses: Deque[int] = field(default_factory=lambda: deque(maxlen=100))
    recent_pdf_failures: Deque[float] = field(default_factory=lambda: deque(maxlen=CIRCUIT_BREAKER_WINDOW))
    circuit_breaker_open: bool = False
    circuit_breaker_open_time: float = 0.0
    memory_usage: float = 0.0
    last_concurrency_adjust: float = field(default_factory=time.time)
    # ... add more as needed
    
    def __str__(self):
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600
        pages_per_hour = self.pages_processed / hours if hours > 0 else 0
        cases_per_hour = self.total_cases / hours if hours > 0 else 0
        avg_response = sum(self.response_times)/len(self.response_times) if self.response_times else 0
        mem = self.memory_usage
        return (f"Pages: {self.pages_processed} ({pages_per_hour:.1f}/hr) | "
                f"Cases: {self.total_cases} ({cases_per_hour:.1f}/hr) | "
                f"PDFs: {self.pdfs_downloaded} | "
                f"Incomplete: {self.incomplete_cases} | Errors: {len(self.errors)} | "
                f"Concurrency: {self.current_concurrency}/{self.active_tasks} | "
                f"Avg Resp: {avg_response:.2f}s | Mem: {mem:.1f}% | "
                f"HTML S/F: {self.html_success}/{self.html_fail} | PDF S/F: {self.pdf_success}/{self.pdf_fail}")

class TaskQueue:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.active_tasks = 0
    
    async def put(self, task):
        await self.queue.put(task)
    
    async def get(self):
        task = await self.queue.get()
        self.active_tasks += 1
        return task
    
    def task_done(self):
        self.active_tasks -= 1
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
logger = logging.getLogger()

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
    """Extract PDF URL and filename from case page using lxml (optimized XPath)"""
    for card in tree.xpath('//div[contains(concat(" ", normalize-space(@class), " "), " card ") and contains(concat(" ", normalize-space(@class), " "), " bg-success ")]'):
        for span in card.xpath('.//span'):
            icon = span.xpath('.//i[contains(concat(" ", normalize-space(@class), " "), " icon-files ")]')
            if icon and 'Download PDF' in span.text_content().strip():
                for a in card.xpath('.//a'):
                    link_text = a.text_content().strip()
                    if link_text.endswith('.pdf'):
                        return a.get('href'), link_text
    return None, None

def extract_case_links(tree) -> List[str]:
    """Extract case detail page links from listing page using lxml (optimized XPath)"""
    return list(dict.fromkeys(
        link.get('href')
        for link in tree.xpath('//div[@class="spost clearfix"]//strong/a[@href]')
    ))

def find_next_page(tree) -> Optional[str]:
    """Find URL for next page in pagination using lxml (optimized XPath)"""
    for a in tree.xpath('//div[contains(concat(" ", normalize-space(@class), " "), " pagging ")]//a[contains(concat(" ", normalize-space(@class), " "), " page-link ")]'):
        if a.text_content().strip() == 'Next':
            return a.get('href')
    return None

def extract_page_number(url: str) -> int:
    """Extract page number from URL"""
    match = re.search(r'/page/(\d+)\.html', url)
    return int(match.group(1)) if match else 1

async def get_tree(client: httpx.AsyncClient, url: str, stats: ScraperStats) -> html.HtmlElement:
    """Fetch and parse HTML page with improved error handling"""
    start_time = time.time()
    logger.debug(f"Fetching page: {url}")
    timeout = adjust_timeouts_based_on_stats(stats)
    try:
        await asyncio.sleep(random.uniform(*POLITENESS_DELAY))
        resp = await client.get(
            url,
            headers=get_random_headers(url),
            timeout=timeout,  # Use dynamic timeout
            follow_redirects=True
        )
        resp.raise_for_status()
        if len(resp.text) < 100:  # Arbitrary minimum
            raise httpx.ReadError("Response too short")
        response_time = time.time() - start_time
        stats.response_times.append(response_time)
        stats.html_success += 1
        stats.recent_statuses.append(resp.status_code)
        logger.info(f"HTML GET {url} {resp.status_code} in {response_time:.2f}s")
        return html.fromstring(resp.text)
    except httpx.ReadTimeout as e:
        logger.warning(f"Read timeout fetching {url} (attempting retry): {e}")
        stats.html_fail += 1
        stats.recent_statuses.append(0)
        raise
    except httpx.TimeoutException as e:
        logger.warning(f"Timeout exception fetching {url}: {e}")
        stats.html_fail += 1
        stats.recent_statuses.append(0)
        raise
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)[:200]}")
        stats.html_fail += 1
        stats.recent_statuses.append(0)
        raise

def sha256sum(filename: str) -> str:
    """Calculate SHA256 checksum of a file"""
    h = hashlib.sha256()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

async def download_pdf(client: httpx.AsyncClient, pdf_url: str, filename: str, stats: ScraperStats) -> str:
    """Download PDF file with retry, resume, and checksum verification"""
    path = os.path.join(PDF_DIR, filename)
    temp_path = path + ".part"
    expected_checksum = None
    for attempt in range(1, MAX_PDF_RETRIES + 1):
        try:
            # Circuit breaker: if open, pause
            if stats.circuit_breaker_open:
                cooldown_left = CIRCUIT_BREAKER_COOLDOWN - (time.time() - stats.circuit_breaker_open_time)
                if cooldown_left > 0:
                    logger.warning(f"PDF circuit breaker open, pausing {cooldown_left:.1f}s")
                    await asyncio.sleep(cooldown_left)
                stats.circuit_breaker_open = False
            # Resume logic
            resume = False
            file_size = 0
            if os.path.exists(temp_path):
                file_size = os.path.getsize(temp_path)
                resume = file_size > 0
            headers = get_random_headers(pdf_url)
            if resume:
                headers['Range'] = f'bytes={file_size}-'
            logger.info(f"Downloading PDF: {pdf_url} (attempt {attempt}, resume={resume})")
            start_time = time.time()
            async with client.stream(
                "GET",
                pdf_url,
                headers=headers,
                timeout=PDF_REQUEST_TIMEOUT
            ) as r:
                r.raise_for_status()
                # If resuming, append to file
                mode = 'ab' if resume else 'wb'
                async with aiofiles.open(temp_path, mode) as f:
                    async for chunk in r.aiter_bytes():
                        await f.write(chunk)
            duration = time.time() - start_time
            logger.info(f"PDF download finished in {duration:.2f}s: {filename}")
            # Checksum verification
            os.replace(temp_path, path)
            checksum = sha256sum(path)
            logger.info(f"PDF checksum: {checksum} for {filename}")
            stats.pdf_success += 1
            stats.recent_pdf_failures.append(0)
            return path
        except Exception as e:
            logger.error(f"Error downloading PDF {pdf_url} (attempt {attempt}): {str(e)[:200]}")
            stats.pdf_fail += 1
            stats.recent_pdf_failures.append(1)
            await asyncio.sleep(min(2 ** attempt, 30))
            # Circuit breaker: if too many failures, open breaker
            if sum(stats.recent_pdf_failures) / len(stats.recent_pdf_failures or [1]) > CIRCUIT_BREAKER_FAILURE_THRESHOLD:
                stats.circuit_breaker_open = True
                stats.circuit_breaker_open_time = time.time()
                logger.warning("PDF circuit breaker triggered!")
    raise Exception(f"Failed to download PDF after {MAX_PDF_RETRIES} attempts: {pdf_url}")

async def parse_case(client: httpx.AsyncClient, case_url: str, stats: ScraperStats) -> Dict:
    """Parse case details from case page using lxml"""
    item = {'url': case_url, 'is_complete': False, 'timestamp': datetime.utcnow().isoformat()}
    
    try:
        logger.debug(f"Parsing case: {case_url}")
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
                item['pdf_path'] = await download_pdf(client, pdf_url, safe_filename, stats)
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
        logger.error(f"Error parsing case {case_url}: {e}")
    
    return item

async def write_jsonl_batch(items: List[Dict], lock: asyncio.Lock):
    """Write batch of items to JSONL file with locking"""
    if not items:
        return
    
    async with lock:
        try:
            async with aiofiles.open(OUTPUT_JSON, 'a', encoding='utf-8') as f:
                for item in items:
                    await f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error writing batch: {e}")

async def worker(case_url: str, client: httpx.AsyncClient, lock: asyncio.Lock, 
                stats: ScraperStats, batch: List[Dict], batch_lock: asyncio.Lock,
                pdf_semaphore: asyncio.Semaphore, existing_case_numbers: Set[str]):
    """Worker function to process a single case with retries"""
    for attempt in range(1, MAX_RETRIES + 1):  # Start from 1
        try:
            # Use exponential backoff for retries
            if attempt > 1:
                delay = calculate_retry_delay(attempt)
                logger.info(f"Retry {attempt} for {case_url} in {delay:.1f}s")
                await asyncio.sleep(delay)
            
            # First check if PDF exists to avoid full parse
            tree = await get_tree(client, case_url, stats)
            pdf_url, pdf_filename = extract_pdf_info(tree)
            
            if pdf_url and pdf_filename:
                safe_filename = sanitize_filename(pdf_filename)
                pdf_path = os.path.join(PDF_DIR, safe_filename)
                if os.path.exists(pdf_path):
                    logger.debug(f"Skipping case with existing PDF: {case_url}")
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
                logger.debug(f"Skipping duplicate case: {case_number}")
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
            if code in (429, 503) or isinstance(e, httpx.TimeoutException):
                logger.warning(f"Attempt {attempt+1} failed for {case_url}, waiting to retry...")
                continue
            async with batch_lock:
                stats.errors.append({
                    'url': case_url,
                    'error': str(e),
                    'type': type(e).__name__,
                    'attempt': attempt,
                    'timestamp': datetime.utcnow().isoformat()
                })
            break
        except Exception as e:
            async with batch_lock:
                stats.errors.append({
                    'url': case_url,
                    'error': str(e),
                    'type': type(e).__name__,
                    'trace': traceback.format_exc(),
                    'attempt': attempt,
                    'timestamp': datetime.utcnow().isoformat()
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
            logger.info(f"Loaded {len(existing)} existing case numbers")
        except Exception as e:
            logger.error(f"Error reading {OUTPUT_JSON}: {e}")
    return existing

async def get_start_page() -> int:
    """Determine starting page from checkpoint file"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                last_page = int(f.read().strip())
                logger.info(f"Resuming from page {last_page + 1}")
                return last_page + 1
        except Exception as e:
            logger.error(f"Error reading checkpoint: {e}")
    return 1

async def update_checkpoint(page_number: int):
    """Update checkpoint file with last completed page"""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            f.write(str(page_number))
    except Exception as e:
        logger.error(f"Error writing checkpoint: {e}")

def check_memory_usage():
    """Check current memory usage"""
    mem = psutil.virtual_memory()
    return mem.percent

async def adjust_concurrency(stats: ScraperStats):
    """Adjust concurrency based on performance, memory, and server health, with gradual changes and logging"""
    while True:
        await asyncio.sleep(CONCURRENCY_ADJUST_INTERVAL)
        mem_usage = check_memory_usage()
        stats.memory_usage = mem_usage
        now = time.time()
        if now - stats.last_concurrency_adjust < CONCURRENCY_ADJUST_INTERVAL:
            continue
        stats.last_concurrency_adjust = now
        avg_response = sum(stats.response_times) / len(stats.response_times) if stats.response_times else 0
        # Only increase concurrency if server is healthy (no recent 429/503, fast HEAD)
        can_increase = True
        if stats.current_concurrency < MAX_CONCURRENCY:
            # Check for recent 429/503
            recent_errors = [s for s in list(stats.recent_statuses)[-20:] if s in (429, 503, 0)]
            if recent_errors:
                can_increase = False
            # Check server health
            if can_increase:
                try:
                    import httpx
                    with httpx.Client() as client:
                        resp = client.head(BASE_URL, timeout=5.0)
                        if resp.status_code != 200:
                            can_increase = False
                except Exception:
                    can_increase = False
        # Gradual adjustment
        old_conc = stats.current_concurrency
        if mem_usage > MEMORY_THRESHOLD and stats.current_concurrency > MIN_CONCURRENCY:
            stats.current_concurrency = max(MIN_CONCURRENCY, stats.current_concurrency - 1)
            logger.warning(f"High memory usage ({mem_usage}%), reducing concurrency to {stats.current_concurrency}")
        elif avg_response > RESPONSE_TIME_TARGET * 1.5 and stats.current_concurrency > MIN_CONCURRENCY:
            stats.current_concurrency = max(MIN_CONCURRENCY, stats.current_concurrency - 1)
            logger.info(f"High response time ({avg_response:.2f}s), reducing concurrency to {stats.current_concurrency}")
        elif avg_response < RESPONSE_TIME_TARGET * 0.7 and stats.current_concurrency < MAX_CONCURRENCY and can_increase:
            stats.current_concurrency = min(MAX_CONCURRENCY, stats.current_concurrency + 1)
            logger.info(f"Low response time ({avg_response:.2f}s), increasing concurrency to {stats.current_concurrency}")
        if stats.current_concurrency != old_conc:
            logger.info(f"Concurrency changed from {old_conc} to {stats.current_concurrency} (mem: {mem_usage:.1f}%, avg_resp: {avg_response:.2f}s)")
        logger.info(f"Current stats: {stats}")

# Replace RETRY_DELAYS with exponential backoff
BASE_RETRY_DELAY = 1.0  # Starting delay in seconds
MAX_RETRY_DELAY = 30.0  # Maximum delay between retries

def calculate_retry_delay(attempt: int) -> float:
    """Calculate exponential backoff with jitter"""
    delay = min(BASE_RETRY_DELAY * (2 ** (attempt - 1)), MAX_RETRY_DELAY)
    return delay * (0.5 + random.random())  # Add jitter

async def check_server_health(client: httpx.AsyncClient) -> bool:
    """Check if server is responsive"""
    try:
        resp = await client.head(BASE_URL, timeout=10.0)
        return resp.status_code == 200
    except Exception:
        return False

def adjust_timeouts_based_on_stats(stats: ScraperStats) -> httpx.Timeout:
    """Adjust timeouts based on recent performance"""
    if len(stats.response_times) < 5:
        return REQUEST_TIMEOUT
    
    avg_response = sum(stats.response_times) / len(stats.response_times)
    if avg_response > READ_TIMEOUT * 0.8:
        return httpx.Timeout(
            connect=CONNECT_TIMEOUT * 1.5,
            read=READ_TIMEOUT * 1.5,
            write=WRITE_TIMEOUT * 1.5,
            pool=POOL_TIMEOUT * 1.5
        )
    return REQUEST_TIMEOUT

async def process_page_batch(client: httpx.AsyncClient, page_batch: List[str], 
                           stats: ScraperStats, existing_case_numbers: Set[str]):
    """Process a batch of pages with enhanced error handling"""
    # Check server health before processing
    if not await check_server_health(client):
        logger.warning("Server unhealthy, pausing for 30s")
        await asyncio.sleep(30)
        return await process_page_batch(client, page_batch, stats, existing_case_numbers)
    
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
                
                logger.info(f"Found {len(filtered_case_urls)} new cases on page {page_url}")
                
                # Create worker tasks with controlled concurrency
                tasks = []
                for url in filtered_case_urls:
                    task = worker(
                        url, client, lock, stats, batch, batch_lock,
                        pdf_semaphore, existing_case_numbers
                    )
                    tasks.append(task)
                
                # Process tasks with semaphore to limit concurrency
                semaphore = asyncio.Semaphore(stats.current_concurrency)
                
                async def run_task(task):
                    async with semaphore:
                        return await task
                
                await asyncio.gather(*[run_task(task) for task in tasks])
                
                stats.pages_processed += 1
                logger.info(f"Progress: {stats}")
                
            except Exception as e:
                logger.error(f"Failed to process page {page_url}: {e}")
                retry_queue.append(page_url)
        
        # Retry failed pages
        if retry_queue:
            logger.info(f"Retrying {len(retry_queue)} failed pages...")
            await process_page_batch(client, list(retry_queue), stats, existing_case_numbers)
        
        # Write any remaining batch items
        if batch:
            await write_jsonl_batch(batch, lock)
        
        logger.info(f"Batch complete. Stats: {stats}")
        
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
    logger.info("Starting optimized scraper")
    stats = ScraperStats()

    # Configure HTTP client with retry
    transport = httpx.AsyncHTTPTransport(
        retries=2,
        limits=httpx.Limits(
            max_connections=MAX_CONCURRENCY * 2,
            max_keepalive_connections=MAX_CONCURRENCY,
            keepalive_expiry=60.0
        ),
        http2=True
    )
    
    async with httpx.AsyncClient(transport=transport) as client:
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
    logger.info(f"Scraping complete. Final stats: {stats}")
    if stats.errors:
        logger.error(f"Encountered {len(stats.errors)} errors")
        for error in stats.errors[:5]:  # Show first 5 errors
            logger.error(f"Error on {error['url']}: {error['error']}")

if __name__ == "__main__":
    import sys
    # --- PROFILING BLOCK START ---
    if '--profile' in sys.argv:
        import cProfile
        import pstats
        with cProfile.Profile() as pr:
            asyncio.run(main())
        stats = pstats.Stats(pr)
        stats.sort_stats('cumtime').print_stats(20)
    else:
        asyncio.run(main())
    # --- PROFILING BLOCK END ---