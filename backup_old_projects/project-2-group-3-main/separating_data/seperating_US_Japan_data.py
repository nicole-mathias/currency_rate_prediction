import pandas as pd


def separate_data(file):

    df = pd.read_csv(file)

    df = pd.DataFrame(df)

    # adding all the attributes relevant to US
    US_data = df[['date','exchange_rate_USD_JY', 'cpi_us', 'inflation_us', 'interest_r_us', 'gdp_pc_us', 'govt_debt_us']]

    Japan_data = df[['date','exchange_rate_USD_JY', 'cpi_j', 'interest_r_j' ,'inflation_j', 'gdp_pc_j', 'govt_debt_j']]

    # saving the data in a csv files
    US_data.to_csv("us_data.csv",index = False)
    Japan_data.to_csv("japan_data.csv", index = False)

    print("-----Separate files for US and Japan were created, filenames: us_data.csv, japan_data.csv-----")




