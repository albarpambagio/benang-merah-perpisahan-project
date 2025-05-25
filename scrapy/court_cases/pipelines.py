# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import json
import os
from scrapy.exceptions import DropItem
from scrapy.pipelines.files import FilesPipeline
import time


class CourtCasesPipeline:
    def process_item(self, item, spider):
        return item

class ValidationPipeline:
    required_fields = ['nomor', 'tanggal_dibacakan', 'lembaga_peradilan']

    def process_item(self, item, spider):
        missing = [f for f in self.required_fields if not item.get(f)]
        if missing:
            item['is_complete'] = False
            item['validation_errors'] = missing
        else:
            item['is_complete'] = True
            item['validation_errors'] = []
        return item

class DuplicatesPipeline:
    def __init__(self):
        self.seen = set()
        # Load existing nomor from output file for persistent deduplication
        output_path = os.path.join(os.path.dirname(__file__), '..', 'court_cases_output.jsonl')
        output_path = os.path.abspath(output_path)
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        nomor = data.get('nomor')
                        if nomor:
                            self.seen.add(nomor)
                    except Exception:
                        continue

    def process_item(self, item, spider):
        nomor = item.get('nomor')
        if nomor and nomor in self.seen:
            raise DropItem(f"Duplicate nomor found: {nomor}")
        if nomor:
            self.seen.add(nomor)
        return item

class JsonWriterPipeline:
    def open_spider(self, spider):
        self.file = open('court_cases_output.jsonl', 'a', encoding='utf-8')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        line = json.dumps(dict(item), ensure_ascii=False) + '\n'
        self.file.write(line)
        return item

class CustomFilesPipeline(FilesPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.logger is automatically set up by Scrapy's base class

    def file_path(self, request, response=None, info=None, *, item=None):
        # Use the sanitized filename from the item if available
        if item and 'file_name' in item:
            return item['file_name']
        return super().file_path(request, response, info, item=item)

    def media_downloaded(self, response, request, info, *, item=None):
        # If the download failed, log and set error
        if response.status != 200:
            if item is not None:
                item['pdf_download_error'] = f"HTTP {response.status}"
            self.logger.warning(f"PDF download failed: {request.url} (status {response.status})")
        return super().media_downloaded(response, request, info, item=item)

    def media_failed(self, failure, request, info, *, item=None):
        # Log and set error on item
        if item is not None:
            item['pdf_download_error'] = str(failure.value)
        self.logger.warning(f"PDF download failed: {request.url} ({failure.value})")
        # Exponential backoff retry logic (up to 3 times)
        retries = request.meta.get('pdf_retry', 0)
        if retries < 3:
            delay = 2 ** retries  # 1s, 2s, 4s
            self.logger.info(f"Retrying PDF download for {request.url} in {delay} seconds (attempt {retries+1})")
            time.sleep(delay)
            new_request = request.replace(dont_filter=True)
            new_request.meta['pdf_retry'] = retries + 1
            return self._enqueue_request(new_request, info)
        return super().media_failed(failure, request, info, item=item)

    def _enqueue_request(self, request, info):
        # Helper to re-enqueue a request for retry
        info.download_func(request, info)
