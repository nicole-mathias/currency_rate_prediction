'''from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key='849436e2473441f19f8072329a33fe94')

# /v2/top-headlines
 top_headlines = newsapi.get_top_headlines(q='bitcoin',
                                          sources='bbc-news,the-verge',
                                          category='business',
                                          language='en',
                                          country='us') 

# /v2/everything
all_articles = newsapi.get_everything(q='financial',
                                      sources='bbc-news,the-verge',
                                      domains='bbc.co.uk,techcrunch.com',
                                      from_param='2008-01-01',
                                      to='2023-09-12',
                                      language='en',
                                      sort_by='relevancy',
                                      country='us'
                                      pageSize = 100)

# /v2/top-headlines/sources
sources = newsapi.get_sources()'''

# importing requests package
import requests	 
import json
import csv
import pandas as pd

def NewsFromBBC():
	
	# BBC news api
	# following query parameters are used
	# source, sortBy and apiKey
	'''query_params = {
	"q":'economy AND US',
    "sources":'bbc-news,the-verge',
    "domains":'bbc.co.uk,techcrunch.com',
    "from_param":'2000-01-01',
    "to":'2023-09-12',
    "language":'en',
    "sort_by":'relevancy',
    "pageSize" : 100,
	"sortBy": "top",
	"apiKey": "4dbc17e007ab436fb66416009dfb59a8"
	}'''
	query_params = {
	"q":'economy AND Japan',
    "sources":'bbc-news,the-verge',
    "domains":'bbc.co.uk,techcrunch.com',
    "from_param":'2000-01-01',
    "to":'2023-09-12',
    "language":'en',
    "sort_by":'relevancy',
    "pageSize" : 100,
	"sortBy": "top",
	"apiKey": "4dbc17e007ab436fb66416009dfb59a8"
	}
	main_url = " https://newsapi.org/v2//everything"

	# fetching data in json format
	res = requests.get(main_url, params=query_params)
	open_bbc_page = res.json()
	with open('data.json', 'w', encoding='utf-8') as f:
		json.dump(open_bbc_page, f, ensure_ascii=False, indent=4)
	print(open_bbc_page)

	# getting all articles in a string article
	article = open_bbc_page["articles"]

	# empty list which will 
	# contain all trending news
	results = []
	years = []

	for ar in article:
		results.append(ar["title"])
	for y in article:
		years.append(y["publishedAt"])

	dict = {'title': results, 'time': years}
	df = pd.DataFrame(dict)
	df.to_csv('Japan.csv')
	#df.to_csv('US.csv')
    
		 

# Driver Code
if __name__ == '__main__':
	
	# function call
	NewsFromBBC() 
