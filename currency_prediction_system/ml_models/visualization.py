import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


'''
Plots - 
1. Time series of exchange rate - Date Vs Exchange Rate
2. Correlation checks - Features Vs Exchange Rate
3. Boxplots - Inflation, CPI, etc
'''


def pair_plots(df):
	'''
	Plots - exchange rate Vs CPI_US, Inflation_US, Inflation_JP
	'''
	fig = make_subplots(rows=2, cols=2)

	fig.add_trace(
		go.Scatter(x=df['inflation_us'], y=df['exchange_rate_USD_JY_x'], mode="markers", opacity=0.8, name="Inflation in US Vs Exchange rate"),
		row=1, col=1
	)

	fig.add_trace(
		go.Scatter(x=df['inflation_j'], y=df['exchange_rate_USD_JY_x'], mode="markers", opacity=0.8, name="Inflation in JP Vs Exchange rate"),
		row=1, col=2
	)

	fig.add_trace(
		go.Scatter(x=df['cpi_us'], y=df['exchange_rate_USD_JY_x'], mode="markers", opacity=0.8, name="CPI US Vs Exchange rate"),
		row=2, col=1
	)

	fig.add_trace(
		go.Scatter(x=df['cpi_j'], y=df['exchange_rate_USD_JY_x'], mode="markers", opacity=0.8, name="CPI JP US Vs Exchange rate"),
		row=2, col=2
	)
	fig['layout']['xaxis']['title']='Inflation (US)'
	fig['layout']['xaxis2']['title']='Inflation (JP)'
	fig['layout']['xaxis3']['title']='CPI-US'
	fig['layout']['xaxis4']['title']='CPI-JP'
	fig['layout']['yaxis']['title']='Exchange Rate (USD-YEN)'
	fig['layout']['yaxis2']['title']='Exchange Rate (USD-YEN)'
	fig['layout']['yaxis3']['title']='Exchange Rate (USD-YEN)'
	fig['layout']['yaxis4']['title']='Exchange Rate (USD-YEN)'

	fig.update_layout(height=1200, width=1500, title_text="Pair Plot wrt Exchange Rate", template='ggplot2')
	fig.show()
	fig.write_image("./pair-plots.png")

def box_plots(df):
	fig = make_subplots(rows=2, cols=2)
	fig.add_trace(go.Box(y=df['inflation_us'], name='Inflation in US', marker_color = 'indianred'), row=1, col=1)
	fig.add_trace(go.Box(y=df['inflation_j'], name = 'Inflation in JP', marker_color = 'lightseagreen'), row=1, col=2)
	fig.add_trace(go.Box(y=df['cpi_us'], name='CPI in US', marker_color = 'indianred'), row=2, col=1)
	fig.add_trace(go.Box(y=df['cpi_j'], name = 'CPI in JP', marker_color = 'lightseagreen'), row=2, col=2)

	fig.update_layout(height=1200, width=1500, title_text="Box Plots", template='ggplot2')
	fig.show()
	fig.write_image("./boxplots.png")

def trendline(df):
	fig = px.scatter(df, x="cpi_us", y="exchange_rate_USD_JY_x", trendline="ols")
	fig.show()

def time_series(df):
	mask = (df.index > '2010-1-1')
	df_new = df.loc[mask]
	

	# Plot
	fig = make_subplots(rows=2, cols=1)

	fig.add_trace(
		go.Scatter(x=df.index, y=df['exchange_rate_USD_JY_x'], opacity=0.8, name="Time Series (1971-01-04 to 2023-09-01)"),
		row=1, col=1
	)

	fig.add_trace(
		go.Scatter(x=df_new.index, y=df_new['exchange_rate_USD_JY_x'], opacity=0.8, name="Time Series (2010-01-01 to 2023-09-01)"),
		row=2, col=1
	)

	fig['layout']['xaxis']['title']='Date'
	fig['layout']['xaxis2']['title']='Date'

	fig['layout']['yaxis']['title']='Exchange Rate (USD-YEN)'
	fig['layout']['yaxis2']['title']='Exchange Rate (USD-YEN)'

	fig.update_layout(height=1000, width=1300, title_text="Time Series Visualization", template='ggplot2', xaxis_title="Date", yaxis_title="Exchange Rate USD-YEN")
	fig.write_html("./time-series.html")
	fig.show()
	fig.write_image("./timeseries.png")


def main():
	df = pd.read_csv("datasets/new_combined_clean.csv")
	df['date'] = pd.to_datetime(df['date'])
	df.set_index('date', inplace=True)
	df.sort_index(ascending=True, inplace=True)
	df = df[df.exchange_rate_USD_JY_x != 156.49025825507422]

	# Call the visualization functions
	pair_plots(df)
	box_plots(df)
	time_series(df) 
	# trendline(df) # Not added to final report

if __name__ == '__main__':
	main()