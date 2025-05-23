# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import json
import os
from scrapy.exceptions import DropItem


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
