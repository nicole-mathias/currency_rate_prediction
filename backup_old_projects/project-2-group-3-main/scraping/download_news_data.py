# pip3 install gnews

import pandas as pd
from gnews import GNews

def download_news(country, query, start_year, end_year):
    all_titles, all_desc, published_at = [], [], []

    for i in range(start_year, end_year):
        google_news = GNews(language='en', country=country, start_date=(i, 1, 1), end_date=(i+1, 1, 1))
        news = google_news.get_news(query)

        for news_data in news:
            print(news_data['title'])
            all_titles += [news_data['title']]
            all_desc += [news_data['description']]
            published_at += [news_data['published date']]

    df = pd.DataFrame({"Title": all_titles, "Description": all_desc, "Published": published_at})

    return df


def main():
    us_news = download_news("US", "financial news US", 2010, 2023)
    jp_news = download_news("JP", "financial news Japan", 2010, 2023)

    us_news.to_csv("./us_news.csv")
    jp_news.to_csv("./japan_news.csv")

if __name__ == '__main__':
    main()
