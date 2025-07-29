import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def prepare_dataset(stockm_file, economic_file):
    stock_data = pd.read_csv(stockm_file)
    economic_data = pd.read_csv(economic_file)

    merged_data = pd.merge(stock_data, economic_data, left_on='Date', right_on='date')

    merged_data['market_movement'] = merged_data.apply(lambda row: 'Up' if row['Close'] >= row['Open'] else 'Down', axis=1)
    # print("merged_data",merged_data)

    X = merged_data.drop(['Date', 'market_movement'], axis=1)
    X_numeric = X.drop(['date'], axis = 1)
    X_string = X['date']
    y = merged_data['market_movement']

    numeric_columns = X_numeric.columns
    
    X_numeric = StandardScaler().fit_transform(X_numeric)
    X_scaled = pd.concat([pd.DataFrame(X_numeric, columns = numeric_columns), X_string], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    y_train_df = pd.DataFrame(y_train, columns=['market_movement'])
    y_test_df = pd.DataFrame(y_test, columns=['market_movement'])
    
    train_df = pd.concat([X_train,y_train_df], axis = 1)
    test_df = pd.concat([X_test,y_test_df], axis = 1)

    csv_file_location = os.getcwd() + "/datasets/categorical_data"

    if not os.path.exists(csv_file_location):
        os.makedirs(csv_file_location)

    train_df.to_csv(csv_file_location + "/train_data.csv", index=False)
    test_df.to_csv(csv_file_location + "/test_data.csv", index = False)


    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = prepare_dataset('datasets/stock_market/S&P_500.csv', 'datasets/new_combined_clean.csv')



