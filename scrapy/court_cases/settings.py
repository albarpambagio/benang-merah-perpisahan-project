# Scrapy settings for court_cases project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = "court_cases"

SPIDER_MODULES = ["court_cases.spiders"]
NEWSPIDER_MODULE = "court_cases.spiders"

ADDONS = {}


# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = "court_cases (+http://www.yourdomain.com)"

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 5

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
DOWNLOAD_DELAY = 1.5
# The download delay setting will honor only one of:
CONCURRENT_REQUESTS_PER_DOMAIN = 5
CONCURRENT_REQUESTS_PER_IP = 8

# Disable cookies (enabled by default)
#COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
#DEFAULT_REQUEST_HEADERS = {
#    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#    "Accept-Language": "en",
#}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    "court_cases.middlewares.CourtCasesSpiderMiddleware": 543,
#}

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
DOWNLOADER_MIDDLEWARES = {
    'court_cases.middlewares.UserAgentRotationMiddleware': 400,
    'court_cases.middlewares.CircuitBreakerMiddleware': 410,
    'court_cases.middlewares.PDFDelayMiddleware': 420,
}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    "scrapy.extensions.telnet.TelnetConsole": None,
#}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
FILES_STORE = 'pdfs'
FILES_URLS_FIELD = 'pdf_url'
FILES_RESULT_FIELD = 'pdfs'
PDF_DIR = 'pdfs'

ITEM_PIPELINES = {
    'court_cases.pipelines.ValidationPipeline': 100,
    'court_cases.pipelines.DuplicatesPipeline': 200,
    'court_cases.spiders.court_cases_spider.CustomFilesPipeline': 400,
    # 'court_cases.pipelines.JsonWriterPipeline': 300,  # Commented out for FilesPipeline
}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True
# The initial download delay
AUTOTHROTTLE_START_DELAY = 1.0
# The maximum download delay to be set in case of high latencies
AUTOTHROTTLE_MAX_DELAY = 30.0
# The average number of requests Scrapy should be sending in parallel to
# each remote server
AUTOTHROTTLE_TARGET_CONCURRENCY = 5.0
# Enable showing throttling stats for every response received:
AUTOTHROTTLE_DEBUG = True

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [500, 502, 503, 504]
HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

# Set settings whose default value is deprecated to a future-proof value
FEED_EXPORT_ENCODING = "utf-8"

# Output (use feed exports for convenience)
FEEDS = {
    'court_cases_output.jsonl': {
        'format': 'jsonlines',
        'encoding': 'utf8',
        'store_empty': False,
        'fields': None,
        'indent': None,
    },
}

# Randomize download delay
RANDOMIZE_DOWNLOAD_DELAY = True

RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]
RETRY_TIMES = 3
RETRY_PRIORITY_ADJUST = -1

# FEED_EXPORT_BATCH_ITEM_COUNT = 1000

# Note: RETRY_DELAY is not a standard Scrapy setting. To implement a custom retry delay, use a custom middleware or adjust DOWNLOAD_DELAY.
# For PDF fallback logic, see the custom pipeline in court_cases_spider.py.

LOG_FILE = 'scrapy.log'
LOG_LEVEL = 'INFO'

DOWNLOAD_TIMEOUT = 90  # Increased timeout for slow PDF downloads
DOWNLOAD_MAXSIZE = 10_000_000  # 10MB max size for large PDFs
