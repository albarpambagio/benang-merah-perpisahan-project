# Mahkamah Agung Scraper (requests + BeautifulSoup)

## Setup

```bash
pip install requests beautifulsoup4
```

## Usage

```bash
python mahkamah_scraper.py
```

- The script will crawl the first N pages of the Mahkamah Agung divorce decisions directory.
- It will extract all requested metadata and download PDFs.
- Output will be saved as JSON and PDFs in a local folder.
