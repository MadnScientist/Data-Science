#!/usr/bin/python
from bs4 import BeautifulSoup
import urllib3
import pandas as pd
import certifi
import os

class Web_scrape():
    def __init__(self, URL):
        if URL is None:
            return None
        
        http = urllib3.PoolManager(cert_reqs = "CERT_REQUIRED", 
					ca_certs = certifi.where())
        response = http.request('GET', URL)
        self.soup = BeautifulSoup(response.data,'html.parser')
        

    def scrape_articles(self):
        if self.soup is None or self.soup == '':
            return []

        ds_article = pd.DataFrame(columns = ["Article_URL", 
					     "Article_Name", 
					     "Article_ImageURL"])
        for i, article in enumerate(self.soup.findAll('a', {"class":"teaser__link"})):
            img = article.find('img', {"class":"component-image__img teaser__img"})
            if img:
                img = img["src"]
            else:
                img = None
            ds_article.loc[i] = [URL + article['href'], article['aria-label'], img]
        return ds_article

if __name__ == "__main__":
    URL = 'https://www.economist.com'
    output_filename = "economist-homepage.csv"
    output_filename = os.getcwd() + '\\' + output_filename
    scrape = Web_scrape(URL)
    if scrape:
        data = scrape.scrape_articles()    
        if len(data) > 0:
            data.to_csv(output_filename)
            print("Web Scraping completed", '\nFile path:' , output_filename)
        else:
            print("No data Found")
    else:
        print("Invalid URL")
