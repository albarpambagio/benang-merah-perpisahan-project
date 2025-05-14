import os

# Base configuration
BASE_URL = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/perceraian.html"
DOWNLOAD_DIR = "data/pdfs"
METADATA_FILE = "data/metadata.csv"
METADATA_NDJSON = "data/metadata.ndjson"
COMPLETED_FILE = "data/completed_cases.txt"

# Scraping limits
MAX_DOCS = 10  # Limit for experimentation
MAX_WORKERS = 4  # Safe concurrency level

# Request settings
MIN_DELAY = 2  # Minimum delay between requests in seconds
MAX_DELAY = 5  # Maximum delay between requests in seconds

# Create necessary directories
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
os.makedirs(os.path.dirname(METADATA_NDJSON), exist_ok=True)
os.makedirs(os.path.dirname(COMPLETED_FILE), exist_ok=True) 