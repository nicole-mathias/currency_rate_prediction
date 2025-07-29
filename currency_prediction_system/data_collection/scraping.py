import requests
from bs4 import BeautifulSoup
import csv



def bbc_scraper(url):

    data = []

    req = requests.get(url)
    b_soup = BeautifulSoup(req.content, 'html5lib') 
    
    # locating the html sections
    section = b_soup.find('article', attrs = {'class': 'ssrcss-pv1rh6-ArticleWrapper e1nh2i2l6'}) 
    page = section.findAll("div", attrs= {'class', 'ssrcss-11r1m41-RichTextComponentWrapper ep2nwvo0'})

    for story in page:
        try:
            item = story.find('p', attrs = {'class' : 'ssrcss-1q0x1qg-Paragraph e1jhz7w10'})
            date = item.get_text().split("-",1)[0]
            content = item.get_text().split("-",1)[1]
            data.append({"year": date, 'news': content})
        except:
            print("Scraping was not done properly!")

    # storing data in a file 
    column_names = ['year','news']

    with open("Japan_scraping.csv",'w') as file:
        writer = csv.DictWriter(file, fieldnames = column_names)
        writer.writeheader()
        writer.writerows(data)



# Url for US data from BBC news
# url = "https://www.bbc.com/news/world-us-canada-16759233"

# url for Japan from BBC news
url = "https://www.bbc.com/news/world-asia-pacific-15219730"
bbc_scraper(url)

