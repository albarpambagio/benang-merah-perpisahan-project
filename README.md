# Court Document Scraper

A robust scraper for Indonesian court documents, built with Scrapling.

## Features

- Polite crawling with randomized delays
- Concurrent processing with safe limits
- Incremental saving and checkpointing
- Automatic retry on failures
- Comprehensive logging
- Metadata extraction and PDF downloads
- Website analysis and estimation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure settings in `config.py`:
- Adjust `MAX_DOCS` for testing/production
- Modify `MIN_DELAY` and `MAX_DELAY` for politeness
- Update `MAX_WORKERS` based on your system

## Usage

### 1. Analyze Website (Optional but Recommended)

Run the estimator to analyze the website and get statistics:
```bash
python scraper/estimator.py
```

This will provide:
- Total number of pages and cases
- Average PDF sizes
- Required storage space
- Estimated completion time
- Recommendations for configuration

### 2. Run the Scraper

Run the scraper:
```bash
python scraper/scraper.py
```

The scraper will:
1. Create necessary directories
2. Load completed URLs from previous runs
3. Fetch case links with pagination
4. Process cases concurrently
5. Save metadata incrementally (NDJSON)
6. Download PDFs
7. Generate CSV summary

## Output

- `data/pdfs/`: Downloaded PDF files
- `data/metadata.ndjson`: Incremental metadata
- `data/metadata.csv`: Complete metadata summary
- `data/completed_cases.txt`: Processed URLs
- `scraper.log`: Detailed operation log
- `estimator.log`: Website analysis log

## Safety Features

- Respects robots.txt
- Random delays between requests
- Limited concurrency
- Automatic retries with exponential backoff
- Graceful error handling
- Checkpointing for resumability

## Notes

- The scraper can be safely interrupted and resumed
- Adjust `MAX_DOCS` for testing
- Monitor `scraper.log` for operation details
- Use the estimator to plan your scraping strategy
