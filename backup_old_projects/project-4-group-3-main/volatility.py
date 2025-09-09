# By looking at the historical data how volatile is the exchange rate?

import plotly.express as px
import pandas as pd
import numpy as np
import os

def volatile():
    data = os.getcwd() + '/datasets/new_combined_clean.csv'

    df = pd.read_csv(data)

    df['Date'] = pd.to_datetime(df['date'])
    df_sorted = df.sort_values(by='Date', ascending=True)

    # Since there is above 40 years of worth data, I have avergaed the monthly data, so 
    # that the graph is not to crowded and it is easier to analyze the movement and volatility.
    df_sorted['Smoothed'] = df_sorted['exchange_rate_USD_JY_x'].rolling(window=30).mean()  # 30-day moving average
    df_sorted['Volatility'] = df_sorted['exchange_rate_USD_JY_x'].rolling(window=30).std()

    plot_data = {
        'Date': df_sorted["Date"],
        'Exchange Rate': df_sorted['Smoothed'],
        'Volatility': df_sorted['Volatility']
    }

    # Create a line chart to show the historical exchange rate vs volatility
    fig = px.line(plot_data, x='Date', y=['Exchange Rate','Volatility'], title='Historical Exchange Rate(JPY wrt US) vs Volatility')

    fig.show()

    # saving the chart as html
    fig.write_html(os.getcwd() + '/plots/volaitility.html')

    # saving the chart as a png
    fig.write_image(os.getcwd() + '/plots/volaitility.png') 

if __name__ == "__main__":
    volatile()
