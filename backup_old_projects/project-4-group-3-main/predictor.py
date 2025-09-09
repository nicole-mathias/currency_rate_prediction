import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor()
}

params = {
    'Ridge': {'alpha': [0.1, 1, 10]},
    'Lasso': {'alpha': [0.1, 1, 10]},
    'RandomForestRegressor': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
    'GradientBoostingRegressor': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
}

def prepare_dataset(file_path):
  df = pd.read_csv(file_path)
  
  df['date'] = pd.to_datetime(df['date'])
  df.sort_values('date', inplace=True)

  X = df[['cpi_us','inflation_us','interest_r_us','gdp_pc_us','govt_debt_us','cpi_j','interest_r_j','inflation_j','gdp_pc_j','govt_debt_j']]
  y = df['exchange_rate_USD_JY_x']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  return X_train_scaled, y_train, X_test_scaled, y_test


def train_models(X_train, y_train):
  best_models = {}

  for model_name in models:
    grid_search = GridSearchCV(models[model_name], params.get(model_name, {}), cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    
    print(f"Best parameters for {model_name} : {grid_search.best_params_}")

  return best_models

def evaluate_model(best_models, X_test, y_test):
  
  for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - MSE: {mse}, MAE: {mae}, R2: {r2}")

    # Plot actual vs predicted values
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted Exchange Rate - {model_name}')
    plt.savefig(f'./plots/{model_name}.png')


def main():
  X_train, y_train, X_test, y_test = prepare_dataset('./datasets/new_combined_clean.csv')
  best_models = train_models(X_train, y_train)
  evaluate_model(best_models, X_test, y_test)

if __name__ == '__main__':
  main()