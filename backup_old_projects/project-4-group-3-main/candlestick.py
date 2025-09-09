# US Stock market changes over the years for top rated equities.
import pandas as pd
import plotly.graph_objects as go
import os


def candle_chart():
    df = pd.read_csv(os.getcwd() + '/datasets/stock_market/S&P_500.csv')
    df = df.sort_values(by='Date', ascending=True)

    # Define the data for the candlestick chart
    open_data = df["Open"]
    high_data = df["High"]
    low_data = df["Low"]
    close_data = df["Close"]
    dates = df["Date"]

    # Create the candlestick chart
    fig = go.Figure(go.Candlestick(
        x=dates,
        open=open_data,
        high=high_data,
        low=low_data,
        close=close_data
    ))

    # Update the layout of the chart
    fig.update_layout(
        title="Candlestick Chart - Daily candlesticks over the years",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis_range=[min(low_data) - 10, max(high_data) + 10],
        xaxis_rangeslider_visible=True
    )
    fig.show()

    # saving the chart as html
    fig.write_html(os.getcwd() + '/plots/US_candlestick.html')

    # saving the chart as a png
    fig.write_image(os.getcwd() + '/plots/US_candlestick.png')

    # displaying another graph with a close-up view of the candle sticks under a shorted period of time
    split_index = int(len(df) * 0.997)
    small_df = df.iloc[split_index:]

    # Define the data for the candlestick chart
    open_data = small_df["Open"]
    high_data = small_df["High"]
    low_data = small_df["Low"]
    close_data = small_df["Close"]
    dates = small_df["Date"]

    # Create the candlestick chart
    fig2 = go.Figure(go.Candlestick(
        x=dates,
        open=open_data,
        high=high_data,
        low=low_data,
        close=close_data
    ))

    fig2.update_layout(
        title="Candlestick Chart - Close up view for a short period of time",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis_range=[min(low_data) - 10, max(high_data) + 10],
        xaxis_rangeslider_visible=True
    )

    fig2.show()

    # saving figure as html
    fig2.write_html(os.getcwd() + '/plots/candle_close_up.html')

    # saving the chart as a png
    fig2.write_image(os.getcwd() + '/plots/candle_close_up.png')

if __name__ == "__main__":
    candle_chart()
