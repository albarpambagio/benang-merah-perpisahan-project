from pathlib import Path

# Base configuration
BASE_URL = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/perceraian.html"
DOWNLOAD_DIR = Path("data/pdfs")
METADATA_NDJSON = Path("data/metadata.ndjson")
COMPLETED_FILE = Path("data/completed_cases.txt")

# Scraping limits
MAX_DOCS = 1000  # Increased for async batch; adjust as needed
MAX_WORKERS = 16  # Higher concurrency for async; tune for your network/server

# Request settings
MIN_DELAY = 2  # Minimum delay between requests in seconds
MAX_DELAY = 5  # Maximum delay between requests in seconds

# Create necessary directories
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
METADATA_NDJSON.parent.mkdir(parents=True, exist_ok=True)
COMPLETED_FILE.parent.mkdir(parents=True, exist_ok=True) 