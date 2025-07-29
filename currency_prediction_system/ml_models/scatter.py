import pandas as pd
import matplotlib.pyplot as plt
import os

path = os.getcwd() + '\\datasets\\new_combined_clean.csv'
# Exchange rate Japanese Yen vs US Dollar
exchange_rate_USD_JY_x = pd.read_csv(path, usecols=['exchange_rate_USD_JY_x'])
# Inflation, consumer prices for Japan
inflation_j = pd.read_csv(path, usecols=['inflation_j'])
# Inflation - US
inflation_us = pd.read_csv(path, usecols=['inflation_us'])
# Consumer price index - US
cpi_us = pd.read_csv(path, usecols=['cpi_us'])
# Consumer price index - Japan
cpi_j = pd.read_csv(path, usecols=['cpi_j'])
# Interest Rates, Discount Rate for Japan
interest_r_j = pd.read_csv(path, usecols=['interest_r_j'])
# Interest Rates for US
interest_r_us = pd.read_csv(path, usecols=['interest_r_us'])
# Constant GDP per capita for Japan
gdp_pc_j = pd.read_csv(path, usecols=['gdp_pc_j'])
# Constant GDP per capita for US
gdp_pc_us = pd.read_csv(path, usecols=['gdp_pc_us'])
# government gross debt - Japan
govt_debt_j = pd.read_csv(path, usecols=['govt_debt_j'])
# government gross debt - US
govt_debt_us = pd.read_csv(path, usecols=['govt_debt_us'])

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

plt.scatter(interest_r_us, exchange_rate_USD_JY_x, color='red')
plt.title('scatter plot 3')
plt.xlabel('Interest Rates, Discount Rate for US')
plt.ylabel('exchange rate')
df = pd.read_csv(path, usecols=['interest_r_us', 'exchange_rate_USD_JY_x'])
correlation = df.corr()['interest_r_us']['exchange_rate_USD_JY_x']
plt.text(interest_r_us.min(), exchange_rate_USD_JY_x.max(),
         f'Correlation: {correlation:.2f}', ha='left', va='top')
plt.show()

plt.scatter(gdp_pc_us, exchange_rate_USD_JY_x, color='orange')
plt.title('scatter plot 4')
plt.xlabel('Constant GDP per capita for US')
plt.ylabel('exchange rate')
df = pd.read_csv(path, usecols=['gdp_pc_us', 'exchange_rate_USD_JY_x'])
correlation = df.corr()['gdp_pc_us']['exchange_rate_USD_JY_x']
plt.text(gdp_pc_us.min(), exchange_rate_USD_JY_x.max(),
         f'Correlation: {correlation:.2f}', ha='left', va='top')
plt.show()

plt.scatter(govt_debt_us, exchange_rate_USD_JY_x, color='purple')
plt.title('scatter plot 5')
plt.xlabel('government gross debt - US')
plt.ylabel('exchange rate')
df = pd.read_csv(path, usecols=['govt_debt_us', 'exchange_rate_USD_JY_x'])
correlation = df.corr()['govt_debt_us']['exchange_rate_USD_JY_x']
plt.text(govt_debt_us.min(), exchange_rate_USD_JY_x.max(),
         f'Correlation: {correlation:.2f}', ha='left', va='top')
plt.show()

plt.scatter(inflation_j, exchange_rate_USD_JY_x, color='olive')
plt.title('scatter plot 6')
plt.xlabel('Inflation - Japan')
plt.ylabel('exchange rate')
df = pd.read_csv(path, usecols=['inflation_j', 'exchange_rate_USD_JY_x'])
correlation = df.corr()['inflation_j']['exchange_rate_USD_JY_x']
plt.text(inflation_j.min(), exchange_rate_USD_JY_x.max(),
         f'Correlation: {correlation:.2f}', ha='left', va='top')
plt.show()

plt.scatter(cpi_j, exchange_rate_USD_JY_x, color='cyan')
plt.title('scatter plot 7')
plt.xlabel('Consumer price index - Japan')
plt.ylabel('exchange rate')
df = pd.read_csv(path, usecols=['cpi_j', 'exchange_rate_USD_JY_x'])
correlation = df.corr()['cpi_j']['exchange_rate_USD_JY_x']
plt.text(cpi_j.min(), exchange_rate_USD_JY_x.max(),
         f'Correlation: {correlation:.2f}', ha='left', va='top')
plt.show()

plt.scatter(interest_r_j, exchange_rate_USD_JY_x, color='pink')
plt.title('scatter plot 8')
plt.xlabel('Interest Rates, Discount Rate for Japan')
plt.ylabel('exchange rate')
df = pd.read_csv(path, usecols=['interest_r_j', 'exchange_rate_USD_JY_x'])
correlation = df.corr()['interest_r_j']['exchange_rate_USD_JY_x']
plt.text(interest_r_j.min(), exchange_rate_USD_JY_x.max(),
         f'Correlation: {correlation:.2f}', ha='left', va='top')
plt.show()

plt.scatter(gdp_pc_j, exchange_rate_USD_JY_x, color='gray')
plt.title('scatter plot 9')
plt.xlabel('Constant GDP per capita for Japan')
plt.ylabel('exchange rate')
df = pd.read_csv(path, usecols=['gdp_pc_j', 'exchange_rate_USD_JY_x'])
correlation = df.corr()['gdp_pc_j']['exchange_rate_USD_JY_x']
plt.text(gdp_pc_j.min(), exchange_rate_USD_JY_x.max(),
         f'Correlation: {correlation:.2f}', ha='left', va='top')
plt.show()

plt.scatter(govt_debt_j, exchange_rate_USD_JY_x, color='yellow')
plt.title('scatter plot 10')
plt.xlabel('government gross debt - Japan')
plt.ylabel('exchange rate')
df = pd.read_csv(path, usecols=['govt_debt_j', 'exchange_rate_USD_JY_x'])
correlation = df.corr()['govt_debt_j']['exchange_rate_USD_JY_x']
plt.text(govt_debt_j.min(), exchange_rate_USD_JY_x.max(),
         f'Correlation: {correlation:.2f}', ha='left', va='top')
plt.show()
