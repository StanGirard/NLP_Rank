import re
from urllib.parse import urljoin, urlsplit, SplitResult
import requests
from bs4 import BeautifulSoup


class RecursiveScraper:
    ''' Scrape URLs in a recursive manner.
    '''
    def __init__(self, url):
        ''' Constructor to initialize domain name and main URL.
        '''
        self.domain = urlsplit(url).netloc
        self.mainurl = url
        self.urls = set()

    def preprocess_url(self, referrer, url):
        ''' Clean and filter URLs before scraping.
        '''
        if not url:
            return None

        fields = urlsplit(urljoin(referrer, url))._asdict() # convert to absolute URLs and split
        fields['path'] = re.sub(r'/$', '', fields['path']) # remove trailing /
        fields['fragment'] = '' # remove targets within a page
        fields = SplitResult(**fields)
        if fields.netloc == self.domain:
            # Scrape pages of current domain only
            cleanurl = ''
            if fields.scheme == 'http':
                cleanurl = fields.geturl()
                cleanurl = cleanurl.replace('http:', 'https:', 1)
            else:
                cleanurl = fields.geturl()
                
            if cleanurl not in self.urls and cleanurl not in self.urls:
                # Return URL only if it's not already in list
                return cleanurl

        return None

    def scrape(self, url=None):
        ''' Scrape the URL and its outward links in a depth-first order.
            If URL argument is None, start from main page.
        '''
        if url is None:
            url = self.mainurl

        print("Scraping {:s} ...".format(url))
        self.urls.add(url)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'lxml')
        for link in soup.findAll("a"):
            childurl = self.preprocess_url(url, link.get("href"))
            if childurl:
                try:
                    self.scrape(childurl)
                except:
                    e = ''


if __name__ == '__main__':
    rscraper = RecursiveScraper("https://www.devlup.fr")
    rscraper.scrape()
    print(rscraper.urls)