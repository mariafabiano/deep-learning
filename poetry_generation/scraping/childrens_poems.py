import re
import requests
import time
from bs4 import BeautifulSoup

def poems():
    base_url = "http://poems.poetrysociety.org.uk/poems/"
    base_page_source = requests.get(base_url).text
    base_soup = BeautifulSoup(base_page_source, "html.parser")
    art_tags = base_soup.find_all(name = 'article')
    hrefs = []
    for art in art_tags:
        hrefs.append(art.a['href'])
    for i, h in enumerate(hrefs):
        time.sleep(1)
        soup = BeautifulSoup(requests.get(h).text, "html.parser")
        d = soup.article.div.text
        if d != '\n\n':
            text_file = open("childrens_poems/poem"+str(i)+".txt", "w")
            text_file.write(d)
            text_file.close()

if __name__ == "__main__":
    poems()
