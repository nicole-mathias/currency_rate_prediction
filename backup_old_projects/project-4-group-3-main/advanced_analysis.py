import numpy as np
import pandas as pd
from transformers import pipeline
from scipy.stats import chi2_contingency


nlp = pipeline("sentiment-analysis")

def get_sentiment(headline):
    result = nlp(headline)[0]
    return result['label']

def eval(merged_df):
    contingency_table = pd.crosstab(merged_df['Sentiment'], merged_df['exchange_rate_change'])

    # Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Phi Coefficient
    phi = (chi2/len(merged_df))**0.5

    print("Chi-Square statistic:", chi2)
    print("P-value:", p)
    print("Phi Coefficient:", phi)

def main():
    news_df = pd.read_csv('./datasets/all_news.csv')
    news_df['Sentiment'] = news_df['Title'].apply(get_sentiment)

    stock_df = pd.read_csv('./datasets/stock_market/S&P_500.csv')
    exchange_df = pd.read_csv('./datasets/new_combined_clean.csv')

    stock_df = stock_df[['date', 'Open', 'Close']]
    exchange_df = exchange_df[['date', 'exchange_rate_USD_JY_x']]

    # Does the market go up or down?
    merged_df = news_df.merge(stock_df, on='date').merge(exchange_df, on='date')
    merged_df['market_movement'] = merged_df['Close'] - merged_df['Open']
    merged_df['market_movement'] = np.where(merged_df['market_movement'] > 0, 'POSITIVE', 'NEGATIVE')

    # Did the exchange rate go up or down?
    merged_df['exchange_rate_change'] = merged_df['exchange_rate_USD_JY_x'].diff()
    merged_df['exchange_rate_change'] = np.where(merged_df['exchange_rate_change'] > 0, 'POSITIVE', 'NEGATIVE')

    eval(merged_df)

if __name__ == '__main__':
    main()
