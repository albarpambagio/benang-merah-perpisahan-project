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
import pandas as pd
import json
import time
import random
import logging
import asyncio
import aiohttp
import aiofiles
from tqdm.asyncio import tqdm as async_tqdm
from typing import Dict, List, Optional
from dataclasses import dataclass
from config import *
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    filename='scraper.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)'
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MyScraper/1.0; +https://yourdomain.com/bot)"
}

@dataclass
class CaseMetadata:
    case_url: str
    pdf_url: Optional[str] = None
    pdf_filename: Optional[str] = None
    def to_dict(self) -> Dict:
        return {
            'case_url': self.case_url,
            'pdf_url': self.pdf_url,
            'pdf_filename': self.pdf_filename
        }
    @classmethod
    def from_dict(cls, data: Dict) -> 'CaseMetadata':
        return cls(
            case_url=data['case_url'],
            pdf_url=data.get('pdf_url'),
            pdf_filename=data.get('pdf_filename')
        )

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

async def fetch_with_retry(session, url, method='get', **kwargs):
    for attempt in range(3):
        try:
            async with session.request(method, url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30), **kwargs) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status} for {url}")
                return await response.text() if method == 'get' else await response.read()
        except Exception as e:
            logging.error(f"Error fetching {url} (attempt {attempt+1}): {e}")
            await asyncio.sleep(min(10, 2 ** attempt + random.uniform(0, 2)))
    raise Exception(f"Failed to fetch {url} after 3 attempts")

async def get_case_links_with_pagination(session, base_url, max_docs):
    case_links = []
    page_num = 1
    next_url = base_url
    pbar = async_tqdm(total=max_docs, desc="Fetching case links")
    while next_url and len(case_links) < max_docs:
        try:
            html = await fetch_with_retry(session, next_url)
            await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            soup = BeautifulSoup(html, 'html.parser')
            new_links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if '/direktori/putusan/' in href and href.endswith('.html') and href not in case_links:
                    new_links.append(urljoin(next_url, href))
                    if len(case_links) + len(new_links) >= max_docs:
                        break
            case_links.extend(new_links)
            await pbar.update(len(new_links))
            logging.info(f"Found {len(new_links)} new links on page {page_num}")
            # Robustly find next page
            next_url = None
            for a in soup.find_all('a', class_='page-link', href=True):
                if a.get('rel') == ['next'] or a.text.strip().lower() == 'next':
                    next_url = urljoin(base_url, a['href'])
                    break
            page_num += 1
        except Exception as e:
            logging.error(f"Error fetching/parsing page {page_num} ({next_url}): {e}")
            break
    await pbar.close()
    logging.info(f"Total case links found: {len(case_links)}")
    return case_links[:max_docs]

async def extract_case_metadata_and_pdf(session, semaphore, case_url):
    metadata = {'case_url': case_url}
    async with semaphore:
        try:
            html = await fetch_with_retry(session, case_url)
            await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            soup = BeautifulSoup(html, 'html.parser')
            # Extract metadata from table
            table = soup.find('table', class_='table')
            if table:
                for tr in table.find_all('tr'):
                    tds = tr.find_all('td')
                    if len(tds) == 2:
                        key = tds[0].get_text(strip=True)
                        value = tds[1].get_text(strip=True)
                        metadata[key] = value
                logging.info(f"Extracted {len(metadata)-1} metadata fields")
            # Extract PDF download link
            pdf_url = None
            for a in soup.find_all('a', href=True):
                href = a['href']
                if '/download_file/' in href and href.endswith('.pdf'):
                    pdf_url = urljoin(case_url, href)
                    break
            metadata['pdf_url'] = pdf_url
            # Download PDF if found
            if pdf_url:
                try:
                    async with session.get(pdf_url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as pdf_response:
                        if pdf_response.status == 200:
                            await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                            pdf_filename = os.path.join(DOWNLOAD_DIR, os.path.basename(pdf_url))
                            async with aiofiles.open(pdf_filename, 'wb') as f:
                                await f.write(await pdf_response.read())
                            metadata['pdf_filename'] = pdf_filename
                            logging.info(f"PDF saved: {pdf_filename}")
                        else:
                            logging.error(f"Error downloading PDF {pdf_url}: HTTP {pdf_response.status}")
                            metadata['pdf_filename'] = None
                except Exception as e:
                    logging.error(f"Error downloading PDF {pdf_url}: {e}")
                    metadata['pdf_filename'] = None
            else:
                logging.warning(f"No PDF found for case: {case_url}")
        except Exception as e:
            logging.error(f"Error processing case {case_url}: {e}")
            raise
    return validate_metadata(metadata).to_dict()

async def process_batch(metadata_list: List[Dict], batch_size: int = 100):
    for i in range(0, len(metadata_list), batch_size):
        batch = metadata_list[i:i + batch_size]
        df = pd.DataFrame(batch)
        batch_file = f"{METADATA_FILE}.batch_{i//batch_size}.csv"
        df.to_csv(batch_file, index=False)
        logging.info(f"Saved batch {i//batch_size} to {batch_file}")

async def main():
    try:
        logging.info("Starting async scraper")
        completed = await load_completed()
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        semaphore = asyncio.Semaphore(MAX_WORKERS)
        async with aiohttp.ClientSession() as session:
            case_links = await get_case_links_with_pagination(session, BASE_URL, MAX_DOCS)
            metadata_list = []
            ndjson_f = await aiofiles.open(METADATA_NDJSON, 'a', encoding='utf-8')
            completed_f = await aiofiles.open(COMPLETED_FILE, 'a', encoding='utf-8')
            tasks = []
            for url in case_links:
                if url not in completed:
                    tasks.append(extract_case_metadata_and_pdf(session, semaphore, url))
            pbar = async_tqdm(total=len(tasks), desc="Processing cases")
            for coro in asyncio.as_completed(tasks):
                try:
                    metadata = await coro
                    metadata_list.append(metadata)
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
            if metadata_list:
                await asyncio.to_thread(process_batch, metadata_list)
                logging.info(f"Processed {len(metadata_list)} records in batches")
        logging.info("Async scraper completed successfully")
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

# Future improvements:
# - Use PlayWrightFetcher or StealthyFetcher for anti-bot
# - Use Scrapling's advanced selectors for unstable or complex sites 