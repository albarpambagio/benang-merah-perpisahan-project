# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class CourtCaseItem(scrapy.Item):
    nomor = scrapy.Field()
    tingkat_proses = scrapy.Field()
    klasifikasi = scrapy.Field()
    kata_kunci = scrapy.Field()
    tahun = scrapy.Field()
    tanggal_register = scrapy.Field()
    lembaga_peradilan = scrapy.Field()
    jenis_lembaga_peradilan = scrapy.Field()
    hakim_ketua = scrapy.Field()
    hakim_anggota = scrapy.Field()
    panitera = scrapy.Field()
    amar = scrapy.Field()
    amar_lainnya = scrapy.Field()
    catatan_amar = scrapy.Field()
    tanggal_musyawarah = scrapy.Field()
    tanggal_dibacakan = scrapy.Field()
    kaidah = scrapy.Field()
    abstrak = scrapy.Field()
    jumlah_view = scrapy.Field()
    jumlah_download = scrapy.Field()
    url = scrapy.Field()
    timestamp = scrapy.Field()
    pdf_url = scrapy.Field()
    file_name = scrapy.Field()
    pdf_filename = scrapy.Field()
    pdf_path = scrapy.Field()
    pdf_download_error = scrapy.Field()
    is_complete = scrapy.Field()
    validation_errors = scrapy.Field()
    parse_error = scrapy.Field()
    traceback = scrapy.Field()
