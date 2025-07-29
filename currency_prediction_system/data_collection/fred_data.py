# pip3 install pyfredapi

import pandas as pd
import pyfredapi as pf

API_KEY = "bb77f6daf770fdf4461dd2500084ab11"

def get_variable(var, name):
    params = {
        "observation_start": "1971-01-01",
        "observation_end": "2023-09-01",
    }

    df = pf.get_series(series_id=var, api_key=API_KEY, **params)
    df = df.rename(columns={"value": name})
    df = df.drop(["realtime_start", "realtime_end"], axis=1)
    return df


def merge_dfs(df1, df2, joinon):
    res = pd.merge(df1.assign(grouper=df1['date'].dt.to_period(joinon)),
        df2.assign(grouper=df2['date'].dt.to_period(joinon)),
        how='left', on='grouper')

    res = res.drop(["date_y", "grouper"], axis=1)
    res = res.rename(columns={"date_x" : "date"})

    return res


df = get_variable("DEXJPUS", "exchange_rate_USD_JY") # Exchange rate Japanese Yen vs US Dollar
df2 = get_variable("JPNCPIALLMINMEI", "cpi_j")       # Consumer price index - Japan
df3 = get_variable("INTDSRJPM193N", "interest_r_j")  # Interest Rates, Discount Rate for Japan 
df4 = get_variable("FPCPITOTLZGJPN", "inflation_j")  # Inflation, consumer prices for Japan
df5 = get_variable("NYGDPPCAPKDJPN", "gdp_pc_j")     # Constant GDP per capita for Japan
df6 = get_variable("CPIAUCSL", "cpi_us")             # Consumer price index - US
df7 = get_variable("FPCPITOTLZGUSA", "inflation_us") # Inflation - US
df8 = get_variable("DFF", "interest_r_us")           #  Federal Funds Effective Rate / Interest rate - US
df9 = get_variable("A939RX0Q048SBEA", "gdp_pc_us")   # GDP per capita -  US
df10 = get_variable("GGGDTAJPA188N", "govt_debt_j")  # government gross debt - Japan 
df11 = get_variable("GFDEGDQ188S", "govt_debt_us")   # Govt gross Debt - US

dfs = [(df2, "M"), 
       (df3, "M"),
       (df4, "Y"), 
       (df5, "Y"),
       (df6, "M"), 
       (df7, "Y"),
       (df8, "D"), 
       (df9, "M"),
       (df10, "Y"), 
       (df11, "M")]

for i, (dfi, joinon) in enumerate(dfs):
    df = merge_dfs(df, dfi, joinon)
    print("Done df ", i)

df.to_csv("./us_jap_data.csv", header=True)