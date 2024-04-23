# "To execute this use scrapy runspider my_crawler.py"
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class MyCrawlerSpider(CrawlSpider):
    name = 'my_crawler'
    allowed_domains = ['en.wikipedia.org']
    start_urls = ['https://en.wikipedia.org/wiki/Main_Page']

    rules = (
        Rule(LinkExtractor(allow='wiki/'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        page_title = response.xpath('//title/text()').get()
        paragraphs = response.xpath('//p/text()').getall()

        if page_title and paragraphs:
            with open('output.txt', 'a', encoding='utf-8') as f:
                f.write(f"Title: {page_title}\n")
                f.write(f"First Paragraph: {paragraphs[0]}\n")
                f.write('\n') 
