"""
Website Analysis Estimator (requests + BeautifulSoup version)

This script analyzes the Indonesian Supreme Court website to estimate:
- Total number of pages
- Total number of cases
- Estimated PDF sizes
- Required storage space
- Estimated completion time
"""

import requests
from bs4 import BeautifulSoup
import logging
import time
import random
from typing import Dict, Tuple
from dataclasses import dataclass
from config import *
import re

# Configure logging
logging.basicConfig(
    filename='estimator.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MyScraper/1.0; +https://yourdomain.com/bot)"
}

@dataclass
class WebsiteStats:
    """Statistics about the website"""
    total_pages: int
    total_cases: int
    avg_cases_per_page: float
    avg_pdf_size_mb: float
    estimated_total_size_gb: float
    estimated_completion_time_hours: float

def get_total_pages(soup):
    """
    Extract the last page number from the pagination links.
    Args:
        soup (BeautifulSoup): Parsed HTML of the page
    Returns:
        int: The last page number
    """
    last_page = 1
    for a in soup.find_all('a', class_='page-link', href=True):
        # Look for the 'Last' link or the highest page number in URLs
        if 'Last' in a.text or 'last' in a.text.lower():
            match = re.search(r'/page/(\d+)\.html', a['href'])
            if match:
                last_page = int(match.group(1))
        else:
            # Fallback: check all page links for the highest number
            match = re.search(r'/page/(\d+)\.html', a['href'])
            if match:
                num = int(match.group(1))
                last_page = max(last_page, num)
    return last_page

def get_page_stats(url: str) -> Tuple[int, int]:
    """
    Get statistics from a single page and robustly detect total pages.
    Args:
        url (str): URL to analyze
    Returns:
        Tuple[int, int]: (number of cases on page, total number of pages)
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Count cases on this page
        case_count = 0
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/direktori/putusan/' in href and href.endswith('.html'):
                case_count += 1

        # Robustly find total pages
        total_pages = get_total_pages(soup)
        return case_count, total_pages
    except Exception as e:
        logging.error(f"Error analyzing page {url}: {e}")
        return 0, 1

def estimate_pdf_size(case_url: str) -> float:
    """
    Estimate PDF size by checking a sample case.
    Args:
        case_url (str): URL of a case page
    Returns:
        float: Estimated PDF size in MB
    """
    try:
        response = requests.get(case_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_url = None
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/download_file/' in href and href.endswith('.pdf'):
                pdf_url = href
                break
        if pdf_url:
            # If the link is relative, join with base
            from urllib.parse import urljoin
            pdf_url = urljoin(case_url, pdf_url)
            head = requests.head(pdf_url, headers=HEADERS, timeout=30)
            size_bytes = int(head.headers.get('content-length', 0))
            return size_bytes / (1024 * 1024)  # MB
    except Exception as e:
        logging.error(f"Error estimating PDF size for {case_url}: {e}")
    return 2.0  # Default estimate if we can't get actual size

def analyze_website() -> WebsiteStats:
    """
    Analyze the website and return statistics.
    Returns:
        WebsiteStats: Website statistics
    """
    logging.info("Starting website analysis (requests + bs4)")
    # Get stats from first page
    cases_per_page, total_pages = get_page_stats(BASE_URL)
    logging.info(f"Found {cases_per_page} cases on first page")
    logging.info(f"Estimated total pages: {total_pages}")
    # Sample a few pages to get average cases per page
    sample_pages = min(5, total_pages)
    total_cases = cases_per_page
    sampled_pages = 1
    for page in range(2, sample_pages + 1):
        try:
            page_url = f"{BASE_URL}?page={page}"
            page_cases, _ = get_page_stats(page_url)
            total_cases += page_cases
            sampled_pages += 1
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        except Exception as e:
            logging.error(f"Error sampling page {page}: {e}")
    avg_cases_per_page = total_cases / sampled_pages if sampled_pages > 0 else 0
    estimated_total_cases = int(avg_cases_per_page * total_pages)
    # Estimate PDF sizes
    sample_cases = min(10, cases_per_page)
    total_pdf_size = 0
    sampled_pdfs = 0
    # Get PDF size estimates from first page cases
    case_links = []
    try:
        response = requests.get(BASE_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/direktori/putusan/' in href and href.endswith('.html'):
                case_links.append(href)
    except Exception as e:
        logging.error(f"Error collecting case links for PDF size estimation: {e}")
    for case_url in case_links[:sample_cases]:
        pdf_size = estimate_pdf_size(case_url)
        total_pdf_size += pdf_size
        sampled_pdfs += 1
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    avg_pdf_size = total_pdf_size / sampled_pdfs if sampled_pdfs > 0 else 2.0
    estimated_total_size = (avg_pdf_size * estimated_total_cases) / 1024  # Convert to GB
    # Estimate completion time
    avg_time_per_case = 3 + (MIN_DELAY + MAX_DELAY) / 2
    estimated_hours = (estimated_total_cases * avg_time_per_case) / (3600 * MAX_WORKERS)
    stats = WebsiteStats(
        total_pages=total_pages,
        total_cases=estimated_total_cases,
        avg_cases_per_page=avg_cases_per_page,
        avg_pdf_size_mb=avg_pdf_size,
        estimated_total_size_gb=estimated_total_size,
        estimated_completion_time_hours=estimated_hours
    )
    return stats

def print_report(stats: WebsiteStats):
    """Print a formatted report of the analysis"""
    print("\nWebsite Analysis Report")
    print("=" * 50)
    print(f"Total Pages: {stats.total_pages:,}")
    print(f"Total Cases: {stats.total_cases:,}")
    print(f"Average Cases per Page: {stats.avg_cases_per_page:.1f}")
    print(f"Average PDF Size: {stats.avg_pdf_size_mb:.1f} MB")
    print(f"Estimated Total Storage: {stats.estimated_total_size_gb:.1f} GB")
    print(f"Estimated Completion Time: {stats.estimated_completion_time_hours:.1f} hours")
    print("=" * 50)
    print("\nRecommendations:")
    print(f"- Set MAX_DOCS to {min(1000, stats.total_cases)} for initial testing")
    print(f"- Ensure at least {stats.estimated_total_size_gb:.1f} GB of free storage")
    print(f"- Plan for approximately {stats.estimated_completion_time_hours:.1f} hours of runtime")
    print("=" * 50)

def main():
    """Main execution function"""
    try:
        logging.info("Starting website analysis")
        stats = analyze_website()
        print_report(stats)
        logging.info("Analysis completed successfully")
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 