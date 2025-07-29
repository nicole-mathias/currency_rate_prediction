# pip3 install sklearn nltk afinn vaderSentiment
import re
import pandas as pd
import nltk
from afinn import Afinn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score

nltk.download('vader_lexicon')

def preprocess_text(text):
    # Remove the news source's name, non alphabets and make it lower case
    text = text.split("- ")[0]
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    text = text.lower()

    return text

def predict_sentiment(df):
    afinn = Afinn()
    vader = SentimentIntensityAnalyzer()

    afinn_scores = [afinn.score(article) for article in df['Title']]
    vader_scores = [vader.polarity_scores(article)['compound'] for article in df['Title']]

    afinn_sentiments = ["positive" if score > 0 else "negative" if score < 0 else "neutral" for score in afinn_scores]
    vader_sentiments = ["positive" if score > 0 else "negative" if score < 0 else "neutral" for score in vader_scores]

    df['afinn_sentiments'] = afinn_sentiments
    df['vader_sentiments'] = vader_sentiments

    return df

def test(preds, labels):
    accuracy = accuracy_score(labels, preds)
    return accuracy

def main():    
    df = pd.read_csv("../datasets/new_news_data/us_news.csv")
    df['Title'] = df['Title'].apply(preprocess_text)

    # Labels for the first 20 rows in US news data
    us_labels = ["positive", "neutral", "neutral", "neutral", "negative",
                     "negative", "neutral", "negative", "negative", "neutral",
                     "negative", "neutral", "neutral", "negative", "negative",
                     "neutral", "negative", "neutral", "positive", "neutral"]

    df = predict_sentiment(df)
    df.to_csv("./us_news_sentiments.csv")
    print(df)

    afinn_preds = df['afinn_sentiments'].tolist()[:20]
    afinn_accuracy = test(afinn_preds, us_labels)
    print("affin Accuracy: ", afinn_accuracy)

    vader_preds = df['vader_sentiments'].tolist()[:20]
    vader_accuracy = test(vader_preds, us_labels)
    print("VADER Accuracy: ", vader_accuracy)

if __name__ == '__main__':
    main()
