#!/usr/bin/env python3
"""
Real Market Model Training
==========================

Train models on actual Federal Reserve USD/JPY data from 1971-2023
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import json
from datetime import datetime

def load_real_market_data():
    """Load real USD/JPY data from Federal Reserve"""
    print("Loading real Federal Reserve USD/JPY data...")
    
    # Load the real market data
    df = pd.read_csv('currency_prediction_system/data_collection/datasets/us_jap_data.csv')
    
    # Clean the data
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Focus on recent data (last 10 years for better relevance)
    recent_data = df[df['date'] >= '2013-01-01'].copy()
    
    print(f"Loaded {len(recent_data)} records from {recent_data['date'].min()} to {recent_data['date'].max()}")
    print(f"Exchange rate range: {recent_data['exchange_rate_USD_JY'].min():.2f} - {recent_data['exchange_rate_USD_JY'].max():.2f}")
    
    return recent_data

def create_real_features(df):
    """Create features from real market data"""
    print("Creating features from real market data...")
    
    features = pd.DataFrame()
    
    # Target variable (next day's exchange rate)
    features['target'] = df['exchange_rate_USD_JY'].shift(-1)
    
    # Current exchange rate
    features['current_rate'] = df['exchange_rate_USD_JY']
    
    # Price changes
    features['daily_change'] = df['exchange_rate_USD_JY'].pct_change()
    features['weekly_change'] = df['exchange_rate_USD_JY'].pct_change(5)
    features['monthly_change'] = df['exchange_rate_USD_JY'].pct_change(20)
    
    # Moving averages
    features['sma_5'] = df['exchange_rate_USD_JY'].rolling(window=5).mean()
    features['sma_20'] = df['exchange_rate_USD_JY'].rolling(window=20).mean()
    features['sma_50'] = df['exchange_rate_USD_JY'].rolling(window=50).mean()
    features['ema_12'] = df['exchange_rate_USD_JY'].ewm(span=12).mean()
    features['ema_26'] = df['exchange_rate_USD_JY'].ewm(span=26).mean()
    
    # Volatility
    features['volatility_5'] = df['exchange_rate_USD_JY'].rolling(window=5).std()
    features['volatility_20'] = df['exchange_rate_USD_JY'].rolling(window=20).std()
    
    # Economic indicators (when available)
    features['us_interest_rate'] = df['interest_r_us'].fillna(method='ffill')
    features['japan_interest_rate'] = df['interest_r_j'].fillna(method='ffill')
    features['us_cpi'] = df['cpi_us'].fillna(method='ffill')
    features['japan_cpi'] = df['cpi_j'].fillna(method='ffill')
    features['us_inflation'] = df['inflation_us'].fillna(method='ffill')
    features['japan_inflation'] = df['inflation_j'].fillna(method='ffill')
    
    # Interest rate differentials
    features['interest_rate_diff'] = features['us_interest_rate'] - features['japan_interest_rate']
    features['inflation_diff'] = features['us_inflation'] - features['japan_inflation']
    
    # Technical indicators
    # RSI
    delta = df['exchange_rate_USD_JY'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    features['macd'] = features['ema_12'] - features['ema_26']
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']
    
    # Bollinger Bands
    features['bb_middle'] = features['sma_20']
    features['bb_upper'] = features['bb_middle'] + (features['volatility_20'] * 2)
    features['bb_lower'] = features['bb_middle'] - (features['volatility_20'] * 2)
    features['bb_position'] = (df['exchange_rate_USD_JY'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
    
    # Lag features
    for i in range(1, 6):
        features[f'rate_lag_{i}'] = df['exchange_rate_USD_JY'].shift(i)
    
    # Remove rows with NaN values
    features = features.dropna()
    
    print(f"Created {len(features.columns)} features")
    print(f"Final dataset: {len(features)} records")
    
    return features

def train_real_models(features):
    """Train models on real market data"""
    print("\nTraining models on real market data...")
    
    # Prepare data
    feature_cols = [col for col in features.columns if col != 'target']
    X = features[feature_cols]
    y = features['target']
    
    # Use time series split for proper validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {}
    results = {}
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    rf_mae = []
    rf_r2 = []
    rf_accuracy = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        
        rf_mae.append(mean_absolute_error(y_test, rf_pred))
        rf_r2.append(r2_score(y_test, rf_pred))
        rf_accuracy.append(calculate_directional_accuracy(y_test, rf_pred))
    
    results['random_forest'] = {
        'mae': np.mean(rf_mae),
        'r2': np.mean(rf_r2),
        'accuracy': np.mean(rf_accuracy)
    }
    
    # XGBoost
    print("Training XGBoost...")
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    
    xgb_mae = []
    xgb_r2 = []
    xgb_accuracy = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        
        xgb_mae.append(mean_absolute_error(y_test, xgb_pred))
        xgb_r2.append(r2_score(y_test, xgb_pred))
        xgb_accuracy.append(calculate_directional_accuracy(y_test, xgb_pred))
    
    results['xgboost'] = {
        'mae': np.mean(xgb_mae),
        'r2': np.mean(xgb_r2),
        'accuracy': np.mean(xgb_accuracy)
    }
    
    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    gb_mae = []
    gb_r2 = []
    gb_accuracy = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        
        gb_mae.append(mean_absolute_error(y_test, gb_pred))
        gb_r2.append(r2_score(y_test, gb_pred))
        gb_accuracy.append(calculate_directional_accuracy(y_test, gb_pred))
    
    results['gradient_boosting'] = {
        'mae': np.mean(gb_mae),
        'r2': np.mean(gb_r2),
        'accuracy': np.mean(gb_accuracy)
    }
    
    return results

def calculate_directional_accuracy(y_true, y_pred):
    """Calculate directional accuracy (percentage of correct direction predictions)"""
    if len(y_true) < 2:
        return 0.0
    
    # Calculate actual and predicted direction changes
    actual_direction = np.diff(y_true) > 0
    predicted_direction = np.diff(y_pred) > 0
    
    # Calculate accuracy
    correct_predictions = np.sum(actual_direction == predicted_direction)
    total_predictions = len(actual_direction)
    
    return (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0

def main():
    """Train real models on actual market data"""
    print("="*60)
    print("REAL MARKET MODEL TRAINING")
    print("="*60)
    print("Training on actual Federal Reserve USD/JPY data (2013-2023)")
    print("="*60)
    
    # Load real data
    df = load_real_market_data()
    
    # Create features
    features = create_real_features(df)
    
    # Train models
    results = train_real_models(features)
    
    # Print results
    print("\n" + "="*60)
    print("REAL MARKET PERFORMANCE RESULTS")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  MAE: {metrics['mae']:.4f} (Average prediction error in JPY)")
        print(f"  R²: {metrics['r2']:.4f} (Explained variance)")
        print(f"  Directional Accuracy: {metrics['accuracy']:.2f}% (Correct direction predictions)")
    
    # Save results
    with open('real_market_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to real_market_results.json")
    print("\n" + "="*60)
    print("NOTE: These are REAL performance metrics from actual market data!")
    print("="*60)
    
    return results

if __name__ == "__main__":
    main() 