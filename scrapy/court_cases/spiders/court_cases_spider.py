import scrapy
from court_cases.items import CourtCaseItem
from lxml import html
from urllib.parse import urljoin
import re
from datetime import datetime
import gc
from scrapy.pipelines.files import FilesPipeline
import os
import time

BASE_URL = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/perceraian.html"
MONTHS = {
    'Januari': '01', 'Februari': '02', 'Maret': '03', 'April': '04',
    'Mei': '05', 'Juni': '06', 'Juli': '07', 'Agustus': '08',
    'September': '09', 'Oktober': '10', 'November': '11', 'Desember': '12'
}

class CourtCasesSpider(scrapy.Spider):
    name = "court_cases"
    allowed_domains = ["putusan3.mahkamahagung.go.id"]
    start_urls = [BASE_URL]
    custom_settings = {
        'DOWNLOAD_TIMEOUT': 60,
        'RETRY_TIMES': 5,
        'CONCURRENT_REQUESTS': 3,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 3,
        'DOWNLOAD_DELAY': 1.5,
        'AUTOTHROTTLE_ENABLED': True,
        'HTTPCACHE_ENABLED': True,
        'FEED_EXPORT_ENCODING': 'utf-8',
        'FILES_STORE': 'pdfs',
        'FILES_URLS_FIELD': 'pdf_url',
        'FILES_RESULT_FIELD': 'pdfs',
        'PDF_DIR': 'pdfs',
    }

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        spider.pdf_dir = crawler.settings.get('PDF_DIR', 'pdfs')
        return spider

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.page_count = 0
        self.case_count = 0
        # self.pdf_dir will be set in from_crawler

    def parse(self, response):
        self.page_count += 1
        self.logger.info(f"Processing page {self.page_count}: {response.url}")
        # Parse case links
        case_links = response.xpath('//div[@class="spost clearfix"]//strong/a/@href').getall()
        self.logger.info(f"Found {len(case_links)} case links on page {self.page_count}")
        if not case_links:
            self.logger.warning(f"No case links found on {response.url}")
        for link in case_links:
            # Set higher priority for case detail pages
            yield response.follow(link, callback=self.parse_case, priority=1)

        # Pagination
        next_page = response.xpath('//div[contains(@class, "pagging")]//a[contains(@class, "page-link") and normalize-space(text())="Next"]/@href').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)
        # For small-scale testing, use CLOSESPIDER_ITEMCOUNT in your Scrapy command.

    def parse_case(self, response):
        self.case_count += 1
        self.logger.info(f"Parsing case page {self.case_count}: {response.url}")
        self.logger.debug(f"parse_case() called for: {response.url}")
        item = CourtCaseItem()
        item['url'] = response.url
        item['timestamp'] = datetime.utcnow().isoformat()
        tree = html.fromstring(response.text)
        # Table data
        table = tree.xpath('//div[contains(@class, "tab-content")]//table[contains(@class, "table")]')
        if table:
            table = table[0]
            for row in table.xpath('.//tr'):
                tds = row.xpath('.//td')
                if len(tds) < 2:
                    continue
                label = tds[0].text_content().strip()
                value = tds[1].text_content().strip()
                if label == 'Nomor':
                    item['nomor'] = value
                elif label == 'Tingkat Proses':
                    item['tingkat_proses'] = value
                elif label == 'Klasifikasi':
                    item['klasifikasi'] = ' '.join(
                        a.text_content().strip() for a in tds[1].xpath('.//a')
                    ) or value
                elif label == 'Kata Kunci':
                    item['kata_kunci'] = value
                elif label == 'Tahun':
                    item['tahun'] = value
                elif label == 'Tanggal Register':
                    item['tanggal_register'] = self.parse_date(value)
                elif label == 'Lembaga Peradilan':
                    item['lembaga_peradilan'] = value
                elif label == 'Jenis Lembaga Peradilan':
                    item['jenis_lembaga_peradilan'] = value
                elif label == 'Hakim Ketua':
                    item['hakim_ketua'] = value
                elif label == 'Hakim Anggota':
                    item['hakim_anggota'] = value
                elif label == 'Panitera':
                    item['panitera'] = value
                elif label == 'Amar':
                    item['amar'] = value
                elif label == 'Amar Lainnya':
                    item['amar_lainnya'] = value
                elif label == 'Catatan Amar':
                    item['catatan_amar'] = tds[1].text_content().strip()
                elif label == 'Tanggal Musyawarah':
                    item['tanggal_musyawarah'] = self.parse_date(value)
                elif label == 'Tanggal Dibacakan':
                    item['tanggal_dibacakan'] = self.parse_date(value)
                elif label == 'Kaidah':
                    item['kaidah'] = value
                elif label == 'Abstrak':
                    item['abstrak'] = value
        # Statistics
        stats_card = tree.xpath('//div[@id="collapseTen"]//div[contains(@class, "card-body")]')
        if stats_card:
            stats_card = stats_card[0]
            view_div = stats_card.xpath('.//div[@title="Jumlah view"]')
            download_div = stats_card.xpath('.//div[@title="Jumlah download"]')
            if view_div:
                item['jumlah_view'] = view_div[0].text_content().strip().replace('\xa0', ' ')
            if download_div:
                item['jumlah_download'] = download_div[0].text_content().strip().replace('\xa0', ' ')
        # PDF
        pdf_url, pdf_filename = self.extract_pdf_info(tree)
        if not pdf_url:
            pdf_url = response.xpath('//a[contains(@href, ".pdf")]/@href').get()
            pdf_filename = response.xpath('//a[contains(@href, ".pdf")]/text()').get()
        if pdf_url:
            pdf_url = response.urljoin(pdf_url)
            item['pdf_url'] = [pdf_url]  # FilesPipeline expects a list
            item['file_name'] = self.sanitize_filename(pdf_filename or os.path.basename(pdf_url))
            item['pdf_filename'] = item['file_name']
            self.logger.info(f"Found PDF: {pdf_url} on {response.url}")
            del tree
            gc.collect()
            yield item
        else:
            self.logger.warning(f"No PDF found on page: {response.url}")
            del tree
            gc.collect()

    def parse_date(self, date_str):
        if not date_str:
            return None
        parts = date_str.strip().split()
        if len(parts) == 3:
            day, month, year = parts
            month = MONTHS.get(month, '01')
            return f"{year}-{month}-{int(day):02d}"
        return date_str

    def extract_pdf_info(self, tree):
        for card in tree.xpath('//div[contains(concat(" ", normalize-space(@class), " ") , " card ") and contains(concat(" ", normalize-space(@class), " ") , " bg-success ")]'):
            for span in card.xpath('.//span'):
                icon = span.xpath('.//i[contains(concat(" ", normalize-space(@class), " ") , " icon-files ")]')
                if icon and 'Download PDF' in span.text_content().strip():
                    for a in card.xpath('.//a'):
                        link_text = a.text_content().strip()
                        if link_text.endswith('.pdf'):
                            return a.get('href'), link_text
        return None, None

    def sanitize_filename(self, filename):
        safe = re.sub(r'[\\/*?:"<>|]', "_", filename)
        safe = safe.replace(' ', '_')
        if not safe.lower().endswith('.pdf'):
            safe += '.pdf'
        return safe 