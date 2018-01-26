import re
import requests
import time
from bs4 import BeautifulSoup

def poems():
    base_url = "http://www.publicdomainpoems.com/"
    base_page_source = requests.get(base_url).text
    base_soup = BeautifulSoup(base_page_source, "html.parser")
    # find all the hrefs for each author
    hrefs = [base_url + ref.a['href'][2:] for ref in base_soup.find_all('li')]
    for i, h1 in enumerate(hrefs[1:-3]):
                    time.sleep(1)
                    poet_soup = BeautifulSoup(requests.get(h1).text, "html.parser")
                    # find all the hrefs for poems associated with the author
                    poem_urls = [(base_url + ref.a['href']).replace(' ', '') for ref in poet_soup.find_all('ul')[1].find_all('li')]
                    for j, h2 in enumerate(poem_urls):
                        print(h2)
                        time.sleep(1)
                        poem_text = requests.get(h2).text.replace('<br/>', '')
                        poem_soup = BeautifulSoup(poem_text, 'html.parser')
                        poem = poem_soup.find_all('div')[3].find_all('h3')[1].next.next
                        if poem != '\n\n':
                            text_file = open("public_domain/poem"+str(i)+'_'+str(j)+".txt", "w")
                            text_file.write(poem)
                            text_file.close()
                        else:
                            print("bad href: {}".format(h2))
    more_poets = hrefs[-2]

if __name__ == "__main__":
    poems()
