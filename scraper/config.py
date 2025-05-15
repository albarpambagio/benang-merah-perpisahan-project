from pathlib import Path

# Base configuration
BASE_URL = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/perceraian.html"
DOWNLOAD_DIR = Path("data/pdfs")
METADATA_NDJSON = Path("data/metadata.ndjson")
COMPLETED_FILE = Path("data/completed_cases.txt")

# Scraping limits
MAX_DOCS = 1000  # Increased for async batch; adjust as needed
MAX_WORKERS = 8  # Reduced from 16

# Request settings
MIN_DELAY = 2  # seconds
MAX_DELAY = 5  # seconds

# Rate limiter settings
RATE_LIMIT_CALLS = 5
RATE_LIMIT_PERIOD = 1

# PDF validation settings
PDF_VALIDATION_TIMEOUT = 30  # seconds to wait for PDF validation
MIN_PDF_PAGES = 1  # minimum number of pages required for valid PDF
MIN_PDF_CONTENT_LENGTH = 100  # minimum characters of text content required

# Batch processing settings
BATCH_TIMEOUT = 3600  # maximum time (seconds) for a batch to run
PARALLEL_BATCH_TIMEOUT = 7200  # maximum time (seconds) for parallel batches
ENABLE_SOUND_NOTIFICATION = True  # whether to play sound when scraping completes

# Metadata validation
REQUIRED_METADATA_FIELDS = [
    'case_url',
    'nomor',
    'tanggal_putus',
    'pengadilan'
]

# Performance settings
BATCH_SIZE = 50  # number of cases to process in memory before writing to disk
MEMORY_CHUNK_SIZE = 1000  # number of cases to process before memory cleanup

# Create necessary directories
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
METADATA_NDJSON.parent.mkdir(parents=True, exist_ok=True)
COMPLETED_FILE.parent.mkdir(parents=True, exist_ok=True) 