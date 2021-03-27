import scrapy

class QuotesSpider(scrapy.Spider):
    name = "quotes-xpath"

    def start_requests(self):
        url = 'http://quotes.toscrape.com/'
        tag = getattr(self, 'tag', None)
        if tag is not None:
            url = url + 'tag/' + tag
        yield scrapy.Request(url, self.parse)

    def parse(self, response):
        for quote in response.xpath("//div[@class='quote']"):
            yield {
                'text': quote.xpath("./span[@class='text']/text()").get(),
                'author': quote.xpath("./span/small[@class='author']/text()").get(),
                'tags': quote.xpath("./div[@class='tags']/a/text()").getall()
            }
        next_page = quote.xpath("//li[@class='next']/a").attrib['href']
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)

class AuthorSpider(scrapy.Spider):
    name = 'author-xpath'

    start_urls = ['http://quotes.toscrape.com/']

    def parse(self, response):
        author_page_links = response.xpath("//a[contains(@href,'author')]")
        yield from response.follow_all(author_page_links, self.parse_author)

        pagination_links = response.xpath("//li[@class='next']/a")
        yield from response.follow_all(pagination_links, self.parse)

    def parse_author(self, response):
        def extract_with_xpath(query):
            return response.xpath(query).get(default='').strip()

        yield {
            'name': extract_with_xpath("//h3[@class='author-title']/text()"),
            'birthdate': extract_with_xpath("//*[@class='author-born-date']/text()"),
            'bio': extract_with_xpath("//*[@class='author-description']/text()")
        }
