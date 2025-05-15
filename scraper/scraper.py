"""
Indonesian Court Document Scraper (asyncio + aiohttp + aiofiles version)

This script scrapes divorce case documents from the Indonesian Supreme Court website asynchronously.
It handles pagination, concurrent processing, and maintains state for resumable scraping.

Key features:
- Polite crawling with randomized delays
- Async concurrent processing with safe limits
- Incremental saving and checkpointing
- Automatic retry on failures
- Progress tracking with tqdm
- Data validation and type safety
- Batch processing for memory efficiency
"""

import os
import json
import time
import random
import logging
import asyncio
import aiohttp
import aiofiles
from tqdm.asyncio import tqdm as async_tqdm
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from config import *
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import psutil
import PyPDF2
from pathlib import Path
import argparse
from concurrent.futures import TimeoutError
import gc
from hashlib import md5

# Use uvloop if available and on supported platform
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    filename='scraper.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)'
)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    # Add more as needed
]

def get_random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

FAILED_PDF_FILE = Path("data/failed_pdfs.txt")

# Memory monitoring helper
MEMORY_LOG_INTERVAL = 100  # log every 100 cases

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    logging.info(f"Memory usage: {mem:.2f} MB")

# PDF validation helper

def is_valid_pdf(filepath: Path) -> bool:
    """
    Validate a PDF file by checking its structure and content.
    
    Args:
        filepath: Path to the PDF file
        
    Returns:
        bool: True if PDF is valid, False otherwise
    """
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            if len(reader.pages) < MIN_PDF_PAGES:
                logging.error(f"PDF has insufficient pages: {len(reader.pages)} < {MIN_PDF_PAGES}")
                return False
            # Check first page has sufficient content
            first_page = reader.pages[0]
            content = first_page.extract_text()
            if not content or len(content.strip()) < MIN_PDF_CONTENT_LENGTH:
                logging.error(f"PDF has insufficient content: {len(content.strip())} < {MIN_PDF_CONTENT_LENGTH}")
                return False
        return True
    except Exception as e:
        logging.error(f"Invalid PDF at {filepath}: {e}")
        return False

@dataclass
class CaseMetadata:
    case_url: str
    register_date: Optional[str] = None
    putus_date: Optional[str] = None
    upload_date: Optional[str] = None
    pdf_url: Optional[str] = None
    pdf_filename: Optional[str] = None
    nomor: Optional[str] = None
    tingkat_proses: Optional[str] = None
    klasifikasi: Optional[str] = None
    kata_kunci: Optional[str] = None
    tahun: Optional[str] = None
    tanggal_register: Optional[str] = None
    lembaga_peradilan: Optional[str] = None
    jenis_lembaga_peradilan: Optional[str] = None
    hakim_ketua: Optional[str] = None
    hakim_anggota: Optional[str] = None
    panitera: Optional[str] = None
    amar: Optional[str] = None
    amar_lainnya: Optional[str] = None
    catatan_amar: Optional[str] = None
    tanggal_musyawarah: Optional[str] = None
    tanggal_dibacakan: Optional[str] = None
    nomor_perkara: Optional[str] = None
    jenis_perkara: Optional[str] = None
    pengadilan: Optional[str] = None
    tanggal_putus: Optional[str] = None
    # Add more as needed

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'CaseMetadata':
        return cls(
            case_url=data.get('case_url'),
            register_date=data.get('register_date'),
            putus_date=data.get('putus_date'),
            upload_date=data.get('upload_date'),
            pdf_url=data.get('pdf_url'),
            pdf_filename=data.get('pdf_filename'),
            nomor=data.get('nomor'),
            tingkat_proses=data.get('tingkat_proses'),
            klasifikasi=data.get('klasifikasi'),
            kata_kunci=data.get('kata_kunci'),
            tahun=data.get('tahun'),
            tanggal_register=data.get('tanggal_register'),
            lembaga_peradilan=data.get('lembaga_peradilan'),
            jenis_lembaga_peradilan=data.get('jenis_lembaga_peradilan'),
            hakim_ketua=data.get('hakim_ketua'),
            hakim_anggota=data.get('hakim_anggota'),
            panitera=data.get('panitera'),
            amar=data.get('amar'),
            amar_lainnya=data.get('amar_lainnya'),
            catatan_amar=data.get('catatan_amar'),
            tanggal_musyawarah=data.get('tanggal_musyawarah'),
            tanggal_dibacakan=data.get('tanggal_dibacakan'),
            nomor_perkara=data.get('nomor_perkara'),
            jenis_perkara=data.get('jenis_perkara'),
            pengadilan=data.get('pengadilan'),
            tanggal_putus=data.get('tanggal_putus'),
            # Add more as needed
        )

# Helper to map table keys to schema fields
TABLE_KEY_MAP = {
    'Nomor': 'nomor',
    'Tingkat Proses': 'tingkat_proses',
    'Klasifikasi': 'klasifikasi',
    'Kata Kunci': 'kata_kunci',
    'Tahun': 'tahun',
    'Tanggal Register': 'tanggal_register',
    'Lembaga Peradilan': 'lembaga_peradilan',
    'Jenis Lembaga Peradilan': 'jenis_lembaga_peradilan',
    'Hakim Ketua': 'hakim_ketua',
    'Hakim Anggota': 'hakim_anggota',
    'Panitera': 'panitera',
    'Amar': 'amar',
    'Amar Lainnya': 'amar_lainnya',
    'Catatan Amar': 'catatan_amar',
    'Tanggal Musyawarah': 'tanggal_musyawarah',
    'Tanggal Dibacakan': 'tanggal_dibacakan',
    'Nomor Perkara': 'nomor_perkara',
    'Jenis Perkara': 'jenis_perkara',
    'Pengadilan': 'pengadilan',
    'Tanggal Putus': 'tanggal_putus',
    # Add more as needed
}

def validate_metadata(metadata: Dict[str, Any]) -> CaseMetadata:
    """
    Validate metadata against required fields.
    
    Args:
        metadata: Dictionary of metadata fields
        
    Returns:
        CaseMetadata: Validated metadata object
        
    Raises:
        ValueError: If required fields are missing
    """
    missing_fields = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata or not metadata[field]]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    return CaseMetadata.from_dict(metadata)

async def load_completed():
    if os.path.exists(COMPLETED_FILE):
        async with aiofiles.open(COMPLETED_FILE, 'r') as f:
            completed = set([line.strip() async for line in f])
        logging.info(f"Loaded {len(completed)} completed URLs from {COMPLETED_FILE}")
    else:
        completed = set()
        logging.info("No completed URLs file found, starting fresh")
    return completed

def get_total_pages(soup):
    last_page = 1
    for a in soup.find_all('a', class_='page-link', href=True):
        if 'Last' in a.text or 'last' in a.text.lower():
            match = re.search(r'/page/(\d+)\.html', a['href'])
            if match:
                last_page = int(match.group(1))
        else:
            match = re.search(r'/page/(\d+)\.html', a['href'])
            if match:
                num = int(match.group(1))
                last_page = max(last_page, num)
    return last_page

# Global rate limiter: 5 requests per second (adjust as needed)
class RateLimiter:
    def __init__(self, max_calls, period):
        self._max_calls = max_calls
        self._period = period
        self._tokens = asyncio.Queue(max_calls)
        for _ in range(max_calls):
            self._tokens.put_nowait(None)
        self._refill_task = None  # Do not start refill here

    async def _refill(self):
        while True:
            await asyncio.sleep(self._period / self._max_calls)
            try:
                self._tokens.put_nowait(None)
            except asyncio.QueueFull:
                pass

    def start_refill(self):
        if self._refill_task is None:
            self._refill_task = asyncio.create_task(self._refill())

    async def acquire(self):
        await self._tokens.get()

# Instantiate a global rate limiter using config values
global_rate_limiter = RateLimiter(max_calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)

# Wrap all network requests with the rate limiter
async def fetch_with_retry(session, url, method='get', **kwargs):
    max_attempts = 5
    base_delay = 5  # Increased from 2
    max_delay = 120  # Increased from 60
    
    for attempt in range(max_attempts):
        try:
            await global_rate_limiter.acquire()
            headers = kwargs.pop('headers', None) or get_random_headers()
            
            # Add more realistic headers
            headers.update({
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "close",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "DNT": "1",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1"
            })
            
            # Add random delay between attempts
            if attempt > 0:
                await asyncio.sleep(random.uniform(1, 3))
            
            async with session.request(method, url, headers=headers, timeout=aiohttp.ClientTimeout(total=30), **kwargs) as response:
                if response.status == 503:
                    # Exponential backoff with jitter and longer base delay
                    delay = min(max_delay, base_delay * (2 ** attempt) + random.uniform(0, 2))
                    logging.warning(f"503 encountered on {url} (attempt {attempt+1}/{max_attempts}), waiting {delay:.1f}s")
                    
                    # Add extra delay for PDF downloads
                    if '/download_file/' in url:
                        delay *= 2  # Double the delay for PDFs
                    
                    await asyncio.sleep(delay)
                    continue
                    
                if response.status != 200:
                    raise Exception(f"HTTP {response.status} for {url}")
                    
                return await response.text() if method == 'get' else await response.read()
                
        except Exception as e:
            logging.error(f"Error fetching {url} (attempt {attempt+1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                delay = min(max_delay, base_delay * (2 ** attempt) + random.uniform(0, 2))
                await asyncio.sleep(delay)
                
    raise Exception(f"Failed to fetch {url} after {max_attempts} attempts")

# Parse page range arguments
parser = argparse.ArgumentParser()
parser.add_argument('--start_page', type=int, default=1)
parser.add_argument('--end_page', type=int, default=1)
args, unknown = parser.parse_known_args()

METADATA_NDJSON = Path("data/metadata.ndjson")
COMPLETED_FILE = Path("data/completed_cases.txt")

# Adaptive rate/delay globals
RATE_LIMIT_CALLS = 10
MIN_DELAY = 1
MAX_DELAY = 2

# Adaptive tuning parameters
recent_errors = []
recent_response_times = []
ADJUST_EVERY = 20  # adjust every 20 requests
ERROR_THRESHOLD = 0.1  # 10% error rate
SLOW_THRESHOLD = 5.0   # 5 seconds average response
MIN_RATE = 2
MAX_RATE = 20
MIN_DELAY_LIMIT = 0.5
MAX_DELAY_LIMIT = 10

def record_result(success, response_time):
    recent_errors.append(0 if success else 1)
    recent_response_times.append(response_time)
    # Keep only the last 100 results
    if len(recent_errors) > 100:
        recent_errors.pop(0)
    if len(recent_response_times) > 100:
        recent_response_times.pop(0)

def auto_tune():
    if len(recent_errors) < ADJUST_EVERY:
        return
    error_rate = sum(recent_errors[-ADJUST_EVERY:]) / ADJUST_EVERY
    avg_time = sum(recent_response_times[-ADJUST_EVERY:]) / ADJUST_EVERY
    global RATE_LIMIT_CALLS, MIN_DELAY, MAX_DELAY
    if error_rate > ERROR_THRESHOLD or avg_time > SLOW_THRESHOLD:
        # Too many errors or too slow: back off
        RATE_LIMIT_CALLS = max(RATE_LIMIT_CALLS - 1, MIN_RATE)
        MIN_DELAY = min(MIN_DELAY + 0.5, MAX_DELAY_LIMIT)
        MAX_DELAY = min(MAX_DELAY + 0.5, MAX_DELAY_LIMIT)
        logging.info(f"Auto-tune: Decreased rate to {RATE_LIMIT_CALLS}, increased delay to {MIN_DELAY}-{MAX_DELAY}")
    else:
        # Doing well: speed up
        RATE_LIMIT_CALLS = min(RATE_LIMIT_CALLS + 1, MAX_RATE)
        MIN_DELAY = max(MIN_DELAY - 0.1, MIN_DELAY_LIMIT)
        MAX_DELAY = max(MAX_DELAY - 0.1, MIN_DELAY_LIMIT)
        logging.info(f"Auto-tune: Increased rate to {RATE_LIMIT_CALLS}, decreased delay to {MIN_DELAY}-{MAX_DELAY}")

# Simple disk cache for HTML responses
CACHE_DIR = ".cache"
def get_cache_key(url):
    return md5(url.encode()).hexdigest()
async def fetch_with_cache(session, url):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = get_cache_key(url)
    cache_path = os.path.join(CACHE_DIR, cache_key)
    if os.path.exists(cache_path):
        async with aiofiles.open(cache_path, 'r', encoding='utf-8') as f:
            return await f.read()
    content = await fetch_with_retry(session, url)
    async with aiofiles.open(cache_path, 'w', encoding='utf-8') as f:
        await f.write(content)
    return content

def get_file_checksum(filepath: Path) -> str:
    """Calculate MD5 checksum of a file."""
    hash_md5 = md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

async def download_pdf_with_retry(session, pdf_url, case_url):
    """Download PDF with retry logic (simple version)."""
    for attempt in range(3):
        try:
            pdf_filename = DOWNLOAD_DIR / Path(pdf_url).name
            async with session.get(pdf_url, headers=get_random_headers(), timeout=aiohttp.ClientTimeout(total=30)) as pdf_response:
                if pdf_response.status == 200:
                    async with aiofiles.open(pdf_filename, 'wb') as f:
                        await f.write(await pdf_response.read())
                    logging.info(f"PDF downloaded: {pdf_filename}")
                    return str(pdf_filename)
                else:
                    logging.error(f"HTTP {pdf_response.status} downloading PDF: {pdf_url}")
                    if attempt < 2:
                        await asyncio.sleep(min(30, 2 ** attempt + random.uniform(0, 2)))
        except Exception as e:
            logging.error(f"Error downloading PDF {pdf_url}: {str(e)}")
            if attempt < 2:
                await asyncio.sleep(min(30, 2 ** attempt + random.uniform(0, 2)))
    return None

# Parallel PDF downloads for a batch
async def download_pdfs(session, metadata_batch):
    tasks = []
    for metadata in metadata_batch:
        if metadata.get('pdf_url') and not metadata.get('pdf_filename'):
            tasks.append(download_pdf_with_retry(session, metadata['pdf_url'], metadata['case_url']))
        else:
            tasks.append(None)
    results = await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)
    # Update metadata with download results
    j = 0
    for i, metadata in enumerate(metadata_batch):
        if metadata.get('pdf_url') and not metadata.get('pdf_filename'):
            result = results[j]
            j += 1
            if not isinstance(result, Exception) and result:
                metadata['pdf_filename'] = result

# Progressive backoff for pagination
async def get_case_links_with_pagination(session, base_url, max_docs, start_page=1, end_page=1):
    case_links = []
    pbar = async_tqdm(total=max_docs, desc="Fetching case links")
    for page in range(start_page, end_page + 1):
        page_url = f"{base_url}?page={page}" if page > 1 else base_url
        # Progressive delay
        page_delay = random.uniform(
            MIN_DELAY + (page/100),
            MAX_DELAY + (page/50)
        )
        await asyncio.sleep(page_delay)
        try:
            start = time.time()
            html = await fetch_with_cache(session, page_url)
            elapsed = time.time() - start
            logging.info(f"Fetched {page_url} in {elapsed:.2f} seconds")
            record_result(True, elapsed)
            soup = BeautifulSoup(html, 'html.parser')
            new_links = []
            # Find all case blocks
            for spost in soup.find_all('div', class_='spost'):
                entry_c = spost.find('div', class_='entry-c')
                if not entry_c:
                    continue
                # Find the case link
                a_tag = entry_c.find('a', href=True)
                if a_tag and '/direktori/putusan/' in a_tag['href'] and a_tag['href'].endswith('.html'):
                    case_url = urljoin(page_url, a_tag['href'])
                    new_links.append({
                        'case_url': case_url
                    })
                    if len(case_links) + len(new_links) >= max_docs:
                        break
            case_links.extend(new_links)
        except Exception as e:
            elapsed = time.time() - start
            record_result(False, elapsed)
            logging.error(f"Error fetching/parsing page {page} ({page_url}): {e}")
            break
        if page % ADJUST_EVERY == 0:
            auto_tune()
        pbar.update(len(new_links))
        logging.info(f"Found {len(new_links)} new links on page {page}")
    pbar.close()
    logging.info(f"Total case links found: {len(case_links)}")
    return case_links[:max_docs]

async def extract_case_metadata_and_pdf(session, semaphore, case_info):
    metadata = CaseMetadata(
        case_url=case_info['case_url']
    )
    async with semaphore:
        try:
            html = await fetch_with_retry(session, case_info['case_url'])
            await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table', class_='table')
            if table:
                for tr in table.find_all('tr'):
                    tds = tr.find_all('td')
                    if len(tds) == 2:
                        key = tds[0].get_text(strip=True)
                        if key == 'Catatan Amar':
                            value = tds[1].get_text(separator='\n', strip=True)
                        else:
                            value = tds[1].get_text(strip=True)
                        field = TABLE_KEY_MAP.get(key)
                        if field:
                            setattr(metadata, field, value)
                logging.info(f"Extracted metadata fields for {metadata.case_url}")
            # Extract PDF download link
            pdf_url = None
            for a in soup.find_all('a', href=True):
                href = a['href']
                if '/download_file/' in href:
                    if '/pdf/' in href or a.text.strip().lower().endswith('.pdf'):
                        pdf_url = urljoin(case_info['case_url'], href)
                        break
            metadata.pdf_url = pdf_url
            # Download PDF if found
            if pdf_url:
                try:
                    pdf_filename = await download_pdf_with_retry(session, pdf_url, case_info['case_url'])
                    if pdf_filename:
                        metadata.pdf_filename = pdf_filename
                        logging.info(f"PDF saved: {pdf_filename}")
                    else:
                        logging.error(f"Failed to download PDF for {case_info['case_url']}")
                        metadata.pdf_filename = None
                        async with aiofiles.open(FAILED_PDF_FILE, 'a', encoding='utf-8') as fail_f:
                            await fail_f.write(f"{case_info['case_url']}\t{pdf_url}\tFailed after 3 attempts\t{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            await fail_f.flush()
                except Exception as e:
                    logging.error(f"Error downloading PDF {pdf_url}: {str(e)}")
                    metadata.pdf_filename = None
                    async with aiofiles.open(FAILED_PDF_FILE, 'a', encoding='utf-8') as fail_f:
                        await fail_f.write(f"{case_info['case_url']}\t{pdf_url}\tException: {str(e)}\t{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        await fail_f.flush()
            else:
                logging.warning(f"No PDF found for case: {case_info['case_url']}")
        except Exception as e:
            logging.error(f"Error processing case {case_info['case_url']}: {str(e)}")
            raise
    return metadata.to_dict()

class StatsTracker:
    def __init__(self):
        self.attempted = 0
        self.success = 0
        self.fail = 0
        self.pdf_success = 0
        self.pdf_fail = 0
        self.start_time = time.time()
        
    def get_summary(self) -> str:
        elapsed = time.time() - self.start_time
        pdf_attempts = self.pdf_success + self.pdf_fail
        pdf_rate = (self.pdf_success / pdf_attempts * 100) if pdf_attempts > 0 else 0.0
        return (
            f"\nScraping complete!\n"
            f"Total cases attempted: {self.attempted}\n"
            f"Case detail fetches succeeded: {self.success}\n"
            f"Case detail fetches failed: {self.fail}\n"
            f"PDF downloads succeeded: {self.pdf_success}\n"
            f"PDF downloads failed: {self.pdf_fail}\n"
            f"Total time: {elapsed:.1f} seconds\n"
            f"Success rate: {(self.success / self.attempted * 100):.1f}%\n"
            f"PDF success rate: {pdf_rate:.1f}%"
        )

# Update process_case_batch_stats to use StatsTracker
async def process_case_batch_stats(session, semaphore, case_batch, completed, stats: StatsTracker):
    tasks = []
    for case_info in case_batch:
        if case_info['case_url'] not in completed:
            tasks.append(extract_case_metadata_and_pdf(session, semaphore, case_info))
    results = []
    async with aiofiles.open(METADATA_NDJSON, 'a', encoding='utf-8') as ndjson_f, \
               aiofiles.open(COMPLETED_FILE, 'a', encoding='utf-8') as completed_f:
        for coro in asyncio.as_completed(tasks):
            try:
                metadata = await coro
                results.append(metadata)
                stats.success += 1
                if metadata.get('pdf_filename'):
                    stats.pdf_success += 1
                elif metadata.get('pdf_url'):
                    stats.pdf_fail += 1
                # Write immediately after each case
                await ndjson_f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
                await completed_f.write(metadata['case_url'] + '\n')
                await ndjson_f.flush()
                await completed_f.flush()
            except Exception as e:
                logging.error(f"Error in async task: {e}")
                stats.fail += 1
    return results

# Update process_with_session_rotation to use StatsTracker
async def process_with_session_rotation(case_links, completed, batch_size, semaphore, stats: StatsTracker):
    for i in range(0, len(case_links), batch_size):
        # Create a new connector for each batch with better settings
        connector = aiohttp.TCPConnector(
            limit=MAX_WORKERS,
            force_close=True,
            enable_cleanup_closed=True,
            ttl_dns_cache=300,
            use_dns_cache=True,
            ssl=False  # Disable SSL verification for better performance
        )
        
        # More realistic headers
        headers = get_random_headers()
        headers.update({
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "close",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1"
        })
        
        # Create session with better timeout settings
        timeout = aiohttp.ClientTimeout(
            total=90,  # Increased from 60
            connect=20,  # Increased from 10
            sock_read=60  # Increased from 30
        )
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers,
            trust_env=True
        ) as session:
            batch = case_links[i:i + batch_size]
            await process_case_batch_stats(session, semaphore, batch, completed, stats)
            
            # Adaptive delay based on success rate with longer minimum delay
            if stats.fail > stats.success * 0.2:  # If more than 20% failures
                delay = random.uniform(15, 25)  # Increased from 10-15
            else:
                delay = random.uniform(8, 15)   # Increased from 5-10
                
            await asyncio.sleep(delay)
            log_memory_usage()
            gc.collect()

# Update main to use StatsTracker
async def main():
    try:
        logging.info("Starting async scraper")
        global_rate_limiter.start_refill()
        completed = await load_completed()
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        semaphore = asyncio.Semaphore(MAX_WORKERS)
        batch_size = BATCH_SIZE * 2
        chunk_size = max(1, MEMORY_CHUNK_SIZE // 2)
        
        # Initialize stats tracker
        stats = StatsTracker()
        
        # Fetch case links
        connector = aiohttp.TCPConnector(
            limit=MAX_WORKERS * 2,
            force_close=True,
            enable_cleanup_closed=True
        )
        async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=60), headers=get_random_headers()) as session:
            case_links = await get_case_links_with_pagination(
                session, BASE_URL, MAX_DOCS, start_page=args.start_page, end_page=args.end_page
            )
        
        # Filter out already completed cases
        case_links = [c for c in case_links if c['case_url'] not in completed]
        if not case_links:
            logging.info("No new cases to process. Exiting.")
            print("No new cases to process. Exiting.")
            return
            
        stats.attempted = len(case_links)
        await process_with_session_rotation(case_links, completed, batch_size, semaphore, stats)
        
        # Print and log summary
        summary = stats.get_summary()
        print(summary)
        logging.info(summary)
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

# Future improvements:
# - Use PlayWrightFetcher or StealthyFetcher for anti-bot
# - Use Scrapling's advanced selectors for unstable or complex sites 