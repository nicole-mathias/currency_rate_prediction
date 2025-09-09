# US Stock market changes over the years for top rated equities - for monthly averages.
import pandas as pd
import plotly.graph_objects as go
import os

def avg_candles():
    df = pd.read_csv(os.getcwd() + '/datasets/stock_market/S&P_500.csv')
    df = df.sort_values(by='Date', ascending=True)

    # seperating the dates, in order to get a monthly average later
    df_date = pd.DataFrame(columns=["Date"])
    df_date['Date'] = pd.to_datetime(df['Date'])
    df_date.set_index('Date', inplace=True)

    # monthly average of dates
    monthly_average = pd.DataFrame(df_date.resample('M').mean())
    df_reset = monthly_average.reset_index().rename(columns={'Date': 'date'})

    # dropping dates from the main dataframe and getting a montly average of all other data.
    df = df.drop(['Date'], axis = 1)

    # The data has been avergaed because, 40 years of daily data was a lot to plot and we wanted to analyze
    # what average monthly data would look like.
    df = df.rolling(window=30).mean()  # 30-day moving average

    # Create the candlestick chart
    fig = go.Figure(go.Candlestick(
        x=df_reset['date'],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    ))

    # Update the layout of the chart
    fig.update_layout(
        title="Candlestick Chart - Monthly Average's over the year",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis_range=[min(df["Low"]) - 10, max(df["High"]) + 10],
        xaxis_rangeslider_visible=True
    )

    # Show the chart
    fig.show()

    # saving the chart as html
    fig.write_html(os.getcwd() + '/plots/mean_candles.html')

    # saving the chart as a png
    fig.write_image(os.getcwd() + '/plots/mean_candles.png') 

if __name__ == "__main__":
    avg_candles()




