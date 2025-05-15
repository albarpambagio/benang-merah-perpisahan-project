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
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from config import *
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import psutil
import PyPDF2
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    filename='scraper.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)'
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MyScraper/1.0; +https://yourdomain.com/bot)"
}

FAILED_PDF_FILE = Path("data/failed_pdfs.txt")

# Memory monitoring helper
MEMORY_LOG_INTERVAL = 100  # log every 100 cases

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    logging.info(f"Memory usage: {mem:.2f} MB")

# PDF validation helper

def is_valid_pdf(filepath):
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            _ = reader.pages  # Try to access pages
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

def validate_metadata(metadata: Dict) -> CaseMetadata:
    if 'case_url' not in metadata:
        raise ValueError("Missing required field: case_url")
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
        asyncio.create_task(self._refill())

    async def _refill(self):
        while True:
            await asyncio.sleep(self._period / self._max_calls)
            try:
                self._tokens.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def acquire(self):
        await self._tokens.get()

# Instantiate a global rate limiter (5 requests per second)
global_rate_limiter = RateLimiter(max_calls=5, period=1)

# Wrap all network requests with the rate limiter
async def fetch_with_retry(session, url, method='get', **kwargs):
    for attempt in range(3):
        try:
            await global_rate_limiter.acquire()
            async with session.request(method, url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30), **kwargs) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status} for {url}")
                return await response.text() if method == 'get' else await response.read()
        except Exception as e:
            logging.error(f"Error fetching {url} (attempt {attempt+1}): {e}")
            await asyncio.sleep(min(10, 2 ** attempt + random.uniform(0, 2)))
    raise Exception(f"Failed to fetch {url} after 3 attempts")

# Parse page range arguments
parser = argparse.ArgumentParser()
parser.add_argument('--start_page', type=int, default=1)
parser.add_argument('--end_page', type=int, default=1)
args, unknown = parser.parse_known_args()

async def get_case_links_with_pagination(session, base_url, max_docs, start_page=1, end_page=1):
    case_links = []
    pbar = async_tqdm(total=max_docs, desc="Fetching case links")
    for page in range(start_page, end_page + 1):
        page_url = f"{base_url}?page={page}" if page > 1 else base_url
        try:
            html = await fetch_with_retry(session, page_url)
            await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            soup = BeautifulSoup(html, 'html.parser')
            new_links = []
            # Find all case blocks
            for spost in soup.find_all('div', class_='spost'):
                entry_c = spost.find('div', class_='entry-c')
                if not entry_c:
                    continue
                # Extract Register, Putus, Upload dates
                small_divs = entry_c.find_all('div', class_='small')
                register_date = putus_date = upload_date = None
                for small in small_divs:
                    text = small.get_text(separator=' ', strip=True)
                    # Look for all three fields in the text
                    reg_match = re.search(r'Register *: *([0-9\-]+)', text)
                    putus_match = re.search(r'Putus *: *([0-9\-]+)', text)
                    upload_match = re.search(r'Upload *: *([0-9\-]+)', text)
                    if reg_match:
                        register_date = reg_match.group(1)
                    if putus_match:
                        putus_date = putus_match.group(1)
                    if upload_match:
                        upload_date = upload_match.group(1)
                # Find the case link
                a_tag = entry_c.find('a', href=True)
                if a_tag and '/direktori/putusan/' in a_tag['href'] and a_tag['href'].endswith('.html'):
                    case_url = urljoin(page_url, a_tag['href'])
                    new_links.append({
                        'case_url': case_url,
                        'register_date': register_date,
                        'putus_date': putus_date,
                        'upload_date': upload_date
                    })
                    if len(case_links) + len(new_links) >= max_docs:
                        break
            case_links.extend(new_links)
            await pbar.update(len(new_links))
            logging.info(f"Found {len(new_links)} new links on page {page}")
        except Exception as e:
            logging.error(f"Error fetching/parsing page {page} ({page_url}): {e}")
            break
    await pbar.close()
    logging.info(f"Total case links found: {len(case_links)}")
    return case_links[:max_docs]

async def extract_case_metadata_and_pdf(session, semaphore, case_info):
    metadata = CaseMetadata(
        case_url=case_info['case_url'],
        register_date=case_info.get('register_date'),
        putus_date=case_info.get('putus_date'),
        upload_date=case_info.get('upload_date')
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
                        # For multi-line/HTML fields, preserve line breaks
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
                if '/download_file/' in href and href.endswith('.pdf'):
                    pdf_url = urljoin(case_info['case_url'], href)
                    break
            metadata.pdf_url = pdf_url
            # Download PDF if found
            if pdf_url:
                try:
                    pdf_filename = DOWNLOAD_DIR / Path(pdf_url).name
                    async with session.get(pdf_url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as pdf_response:
                        if pdf_response.status == 200:
                            await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                            async with aiofiles.open(pdf_filename, 'wb') as f:
                                await f.write(await pdf_response.read())
                            # Validate PDF
                            if is_valid_pdf(pdf_filename):
                                metadata.pdf_filename = str(pdf_filename)
                                logging.info(f"PDF saved and validated: {pdf_filename}")
                            else:
                                metadata.pdf_filename = None
                                logging.error(f"Downloaded file is not a valid PDF: {pdf_filename}")
                                async with aiofiles.open(FAILED_PDF_FILE, 'a', encoding='utf-8') as fail_f:
                                    await fail_f.write(f"{case_info['case_url']}\t{pdf_url}\tInvalid PDF\n")
                                    await fail_f.flush()
                        else:
                            logging.error(f"Error downloading PDF {pdf_url}: HTTP {pdf_response.status}")
                            metadata.pdf_filename = None
                            # Record failed PDF
                            async with aiofiles.open(FAILED_PDF_FILE, 'a', encoding='utf-8') as fail_f:
                                await fail_f.write(f"{case_info['case_url']}\t{pdf_url}\tHTTP {pdf_response.status}\n")
                                await fail_f.flush()
                except Exception as e:
                    logging.error(f"Error downloading PDF {pdf_url}: {e}")
                    metadata.pdf_filename = None
                    # Record failed PDF
                    async with aiofiles.open(FAILED_PDF_FILE, 'a', encoding='utf-8') as fail_f:
                        await fail_f.write(f"{case_info['case_url']}\t{pdf_url}\t{e}\n")
                        await fail_f.flush()
            else:
                logging.warning(f"No PDF found for case: {case_info['case_url']}")
        except Exception as e:
            logging.error(f"Error processing case {case_info['case_url']}: {e}")
            raise
    return metadata.to_dict()

async def main():
    try:
        logging.info("Starting async scraper")
        completed = await load_completed()
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        semaphore = asyncio.Semaphore(MAX_WORKERS)
        async with aiohttp.ClientSession() as session:
            case_links = await get_case_links_with_pagination(
                session, BASE_URL, MAX_DOCS, start_page=args.start_page, end_page=args.end_page
            )
            metadata_list = []
            ndjson_f = await aiofiles.open(METADATA_NDJSON, 'a', encoding='utf-8')
            completed_f = await aiofiles.open(COMPLETED_FILE, 'a', encoding='utf-8')
            tasks = []
            for idx, case_info in enumerate(case_links):
                if case_info['case_url'] not in completed:
                    tasks.append(extract_case_metadata_and_pdf(session, semaphore, case_info))
                # Memory monitoring
                if (idx + 1) % MEMORY_LOG_INTERVAL == 0:
                    log_memory_usage()
            pbar = async_tqdm(total=len(tasks), desc="Processing cases")
            for coro in asyncio.as_completed(tasks):
                try:
                    metadata = await coro
                    metadata_list.append(metadata)
                    # Incremental metadata consistency: write and flush after each case
                    await ndjson_f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
                    await ndjson_f.flush()
                    await completed_f.write(metadata['case_url'] + '\n')
                    await completed_f.flush()
                    await pbar.update(1)
                except Exception as e:
                    logging.error(f"Error in async task: {e}")
            await pbar.close()
            await ndjson_f.close()
            await completed_f.close()
        logging.info("Async scraper completed successfully")
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

# Future improvements:
# - Use PlayWrightFetcher or StealthyFetcher for anti-bot
# - Use Scrapling's advanced selectors for unstable or complex sites 