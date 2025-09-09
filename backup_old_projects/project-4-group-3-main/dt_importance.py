import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px
import os


def dt_regressor():
    # Load training data from the CSV file
    train_file_path = os.getcwd() + '/datasets/categorical_data/train_data.csv'
    train_df = pd.read_csv(train_file_path)

    # Load testing data from the CSV file
    test_file_path = os.getcwd() + '/datasets/categorical_data/test_data.csv'
    test_df = pd.read_csv(test_file_path)

    # Dropping features which are not important 
    # Note: The below features dropped are from prior analysis using the same code, to make things simple
    # I have already dropped non-important features, which can be seen below.
    X_train = train_df.drop(['market_movement','exchange_rate_USD_JY_y','exchange_rate_USD_JY_x','date','Volume','Low','Open','gdp_pc_j','Adj Close','interest_r_j','govt_debt_j','inflation_j','gdp_pc_us','govt_debt_us'], axis=1)
    y_train = train_df['exchange_rate_USD_JY_x']

    X_test = test_df.drop(['market_movement','exchange_rate_USD_JY_y','exchange_rate_USD_JY_x','date','Volume','Low','Open','gdp_pc_j','Adj Close','interest_r_j','govt_debt_j','inflation_j','gdp_pc_us','govt_debt_us'], axis=1)
    y_test = test_df['exchange_rate_USD_JY_x']

    # Create a decision tree regressor
    regressor = DecisionTreeRegressor(random_state=42)

    # Perform grid search for hyperparameter tuning
    param_grid = {
        'criterion': ['friedman_mse','absolute_error','squared_error'],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
    }

    grid_search = GridSearchCV(regressor, param_grid, cv=10)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Train a decision tree regressor with the best hyperparameters
    best_regressor = DecisionTreeRegressor(**best_params)
    best_regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = best_regressor.predict(X_test)

    # Evaluate the model using mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}\n")

    results = {}
    # Display feature importance
    feature_importance = best_regressor.feature_importances_
    for i, (feature, importance) in enumerate(zip(X_train.columns, feature_importance)):
        print(f"Feature {i + 1}: {feature} - Importance: {importance}")
        results[feature] = importance

    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

    print("\n----------------Importance score in sorted order----------")
    print(results)

    data = {
        'Features': list(results.keys()),
        'Importance': list(results.values())
    }

    # creating bar chart using plotly
    fig = px.bar(data, x='Features', y='Importance', title='Important Features for Currency Exchange Rates (Decision Tree)')
    fig.show()

    # saving the chart into an html file
    fig.write_html(os.getcwd() + '/plots/dt_importance.html')
    
    # saving the chart as a png
    fig.write_image(os.getcwd() + '/plots/dt_importance.png') 



if __name__ == "__main__":
    dt_regressor()

