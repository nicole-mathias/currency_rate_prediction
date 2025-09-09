import pandas as pd
import matplotlib.pyplot as plt
import os
base_path = os.getcwd()
path = base_path.replace('plot', 'datasets/new_combined_clean.csv')
# Exchange rate Japanese Yen vs US Dollar
exchange_rate_USD_JY_x = pd.read_csv(path, usecols=['exchange_rate_USD_JY_x'])
# Inflation, consumer prices for Japan
inflation_j = pd.read_csv(path, usecols=['inflation_j'])
# Inflation - US
inflation_us = pd.read_csv(path, usecols=['inflation_us'])
# Consumer price index - US
cpi_us = pd.read_csv(path, usecols=['cpi_us'])
# Interest Rates, Discount Rate for Japan
interest_r_j = pd.read_csv(path, usecols=['interest_r_j'])
# Interest Rates for US
interest_r_us = pd.read_csv(path, usecols=['interest_r_us'])
# draw histogram

plt.hist(interest_r_j, bins=50, alpha=0.5, label='interest_r_j')
plt.hist(interest_r_us, bins=50, alpha=0.5, label='interest_r_us')
plt.hist(inflation_us, bins=50, alpha=0.5, label='inflation_j')
plt.hist(inflation_j, bins=50, alpha=0.5, label='inflation_us')

plt.title('histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.hist(interest_r_j, bins=50, alpha=0.5, label='interest_r_j')
plt.hist(inflation_us, bins=50, alpha=0.5, label='inflation_j')
plt.title('histogram Japan')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.hist(interest_r_us, bins=50, alpha=0.5, label='interest_r_us')
plt.hist(inflation_j, bins=50, alpha=0.5, label='inflation_us')
plt.title('histogram US')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# draw scatter plot
plt.scatter(inflation_us, exchange_rate_USD_JY_x, color='blue')
plt.title('scatter plot 1')
plt.xlabel('Inflation - US')
plt.ylabel('exchange rate')
df = pd.read_csv(path, usecols=['inflation_us', 'exchange_rate_USD_JY_x'])
correlation = df.corr()['inflation_us']['exchange_rate_USD_JY_x']
plt.text(inflation_us.min(), exchange_rate_USD_JY_x.max(),
         f'Correlation: {correlation:.2f}', ha='left', va='top')
plt.show()
plt.scatter(cpi_us, exchange_rate_USD_JY_x, color='green')
plt.title('scatter plot 2')
plt.xlabel('Consumer price index - US')
plt.ylabel('exchange rate')
df = pd.read_csv(path, usecols=['cpi_us', 'exchange_rate_USD_JY_x'])
correlation = df.corr()['cpi_us']['exchange_rate_USD_JY_x']
plt.text(cpi_us.min(), exchange_rate_USD_JY_x.max(),
         f'Correlation: {correlation:.2f}', ha='left', va='top')
plt.show()
plt.scatter(interest_r_j, exchange_rate_USD_JY_x, color='red')
plt.title('scatter plot 3')
plt.xlabel('Interest Rates, Discount Rate for Japan')
plt.ylabel('exchange rate')
df = pd.read_csv(path, usecols=['interest_r_j', 'exchange_rate_USD_JY_x'])
correlation = df.corr()['interest_r_j']['exchange_rate_USD_JY_x']
plt.text(interest_r_j.min(), exchange_rate_USD_JY_x.max(),
         f'Correlation: {correlation:.2f}', ha='left', va='top')
plt.show()
