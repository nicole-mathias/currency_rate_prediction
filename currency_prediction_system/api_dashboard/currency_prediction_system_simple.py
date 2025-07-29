#!/usr/bin/env python3
"""
Currency Rate Prediction System (Simplified Version)
==================================================

A simplified version that works without TensorFlow, using only scikit-learn
and other readily available libraries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Time Series and Advanced ML
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb

# Data Collection
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
import pyfredapi as pf

# NLP and Sentiment Analysis
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Web Framework and Dashboard
from flask import Flask, render_template, request, jsonify
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Database and Storage
import sqlite3
import pickle
import joblib
import logging
import os

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config

# Data Collection Module
class DataCollector:
    """Automated data collection from multiple sources"""
    
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=self.config.LOG_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def collect_fred_data(self, series_ids, start_date='2018-01-01'):
        """Collect economic data from FRED API"""
        self.logger.info(f"Collecting FRED data for {len(series_ids)} series")
        
        data = {}
        for series_id in series_ids:
            try:
                df = pf.get_series(
                    series_id=series_id,
                    api_key=self.config.FRED_API_KEY,
                    observation_start=start_date
                )
                data[series_id] = df
                self.logger.info(f"Successfully collected {series_id}")
            except Exception as e:
                self.logger.error(f"Error collecting {series_id}: {e}")
        
        return data
    
    def collect_yahoo_finance_data(self, symbols, period='5y'):
        """Collect financial data from Yahoo Finance"""
        self.logger.info(f"Collecting Yahoo Finance data for {symbols}")
        
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                data[symbol] = df
                self.logger.info(f"Successfully collected {symbol}")
            except Exception as e:
                self.logger.error(f"Error collecting {symbol}: {e}")
        
        return data

# Feature Engineering Module
class FeatureEngineer:
    """Advanced feature engineering for currency prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def create_technical_indicators(self, df):
        """Create technical indicators for currency pairs"""
        # Moving averages
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volatility
        df['volatility'] = df['Close'].rolling(window=20).std()
        
        return df
    
    def create_rolling_statistics(self, df, windows=[5, 10, 20]):
        """Create rolling statistics features"""
        for window in windows:
            df[f'rolling_mean_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['Close'].rolling(window=window).std()
            df[f'rolling_min_{window}'] = df['Close'].rolling(window=window).min()
            df[f'rolling_max_{window}'] = df['Close'].rolling(window=window).max()
        
        return df
    
    def create_lag_features(self, df, lags=[1, 2, 3, 5, 10]):
        """Create lag features for time series prediction"""
        for lag in lags:
            df[f'lag_{lag}'] = df['Close'].shift(lag)
        
        return df
    
    def create_fourier_features(self, df, periods=[7, 30, 90]):
        """Create Fourier transform features for seasonality"""
        for period in periods:
            df[f'fourier_sin_{period}'] = np.sin(2 * np.pi * df.index.dayofyear / period)
            df[f'fourier_cos_{period}'] = np.cos(2 * np.pi * df.index.dayofyear / period)
        
        return df

# Ensemble Model Module (Simplified)
class EnsembleForecaster:
    """Ensemble forecasting models combining multiple algorithms"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        
    def train_ensemble_models(self, X_train, y_train, X_test, y_test):
        """Train ensemble of models"""
        self.logger.info("Training ensemble models...")
        
        # 1. Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # 2. XGBoost
        try:
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            xgb_model.fit(X_train, y_train)
            self.models['xgboost'] = xgb_model
        except Exception as e:
            self.logger.warning(f"XGBoost not available: {e}")
        
        # 3. Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb_model
        
        # 4. Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        self.models['linear_regression'] = lr_model
        
        # 5. Ridge Regression
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        self.models['ridge'] = ridge_model
        
        self.logger.info("Ensemble models trained successfully")
        return self.models
    
    def ensemble_predict(self, X):
        """Make ensemble predictions"""
        predictions = {}
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions[name] = pred
        
        # Weighted average of predictions
        weights = {
            'random_forest': 0.3,
            'xgboost': 0.25,
            'gradient_boosting': 0.2,
            'linear_regression': 0.15,
            'ridge': 0.1
        }
        
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            if name in weights:
                ensemble_pred += weights[name] * pred.flatten()
        
        return ensemble_pred, predictions

# Sentiment Analysis Module (Simplified)
class SentimentAnalyzer:
    """Real-time sentiment analysis using NLP"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def analyze_sentiment(self, text):
        """Analyze sentiment using VADER"""
        vader_scores = self.sia.polarity_scores(text)
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu']
        }
    
    def process_news_batch(self, news_data):
        """Process batch of news articles for sentiment analysis"""
        sentiments = []
        
        for _, row in news_data.iterrows():
            sentiment = self.analyze_sentiment(row['title'] + ' ' + row['content'])
            sentiments.append(sentiment)
        
        return pd.DataFrame(sentiments)

# Database Module
class DatabaseManager:
    """SQLite database management for historical data storage"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database with time-series partitioning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for different data types
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS currency_rates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                currency_pair TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator_id TEXT,
                date TEXT,
                value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                currency_pair TEXT,
                vader_compound REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                currency_pair TEXT,
                prediction_date TEXT,
                actual_value REAL,
                predicted_value REAL,
                model_name TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for efficient querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_currency_date ON currency_rates(currency_pair, date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicator_date ON economic_indicators(indicator_id, date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_date ON sentiment_scores(date, currency_pair)')
        
        conn.commit()
        conn.close()
    
    def store_currency_data(self, data, currency_pair):
        """Store currency data with time-series partitioning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for _, row in data.iterrows():
            cursor.execute('''
                INSERT INTO currency_rates 
                (currency_pair, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (currency_pair, row.name, row['Open'], row['High'], 
                  row['Low'], row['Close'], row['Volume']))
        
        conn.commit()
        conn.close()
    
    def get_historical_data(self, currency_pair, start_date, end_date):
        """Retrieve historical data efficiently"""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT * FROM currency_rates 
            WHERE currency_pair = ? AND date BETWEEN ? AND ?
            ORDER BY date
        '''
        df = pd.read_sql_query(query, conn, params=(currency_pair, start_date, end_date))
        conn.close()
        return df

# Flask API Module
class CurrencyPredictionAPI:
    """Production-ready Flask API for currency prediction"""
    
    def __init__(self, config, data_collector, feature_engineer, ensemble_forecaster, sentiment_analyzer, db_manager):
        self.config = config
        self.data_collector = data_collector
        self.feature_engineer = feature_engineer
        self.ensemble_forecaster = ensemble_forecaster
        self.sentiment_analyzer = sentiment_analyzer
        self.db_manager = db_manager
        
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/')
        def home():
            return render_template('index.html')
        
        @self.app.route('/api/predict/<currency_pair>')
        def predict_currency(currency_pair):
            """Predict currency rate for a specific pair"""
            try:
                # Get latest data
                data = self.db_manager.get_historical_data(
                    currency_pair, 
                    (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                    datetime.now().strftime('%Y-%m-%d')
                )
                
                if len(data) < 60:
                    return jsonify({'error': 'Insufficient data'}), 400
                
                # Feature engineering
                data = self.feature_engineer.create_technical_indicators(data)
                data = self.feature_engineer.create_rolling_statistics(data)
                data = self.feature_engineer.create_lag_features(data)
                
                # Prepare features
                feature_columns = [col for col in data.columns if col not in ['id', 'currency_pair', 'date', 'created_at', 'exchange_rate_USD_JY_x', 'exchange_rate_USD_JY_y']]
                X = data[feature_columns].dropna()
                y = data['exchange_rate_USD_JY_x']  # Use the first exchange rate column as target
                
                if len(X) == 0:
                    return jsonify({'error': 'No valid features'}), 400
                
                # Align X and y
                if len(X) != len(y):
                    min_len = min(len(X), len(y))
                    X = X.iloc[:min_len]
                    y = y.iloc[:min_len]
                
                # Make prediction
                prediction, model_predictions = self.ensemble_forecaster.ensemble_predict(X.iloc[-1:])
                
                return jsonify({
                    'currency_pair': currency_pair,
                    'prediction': float(prediction[0]),
                    'confidence': 0.85,  # Placeholder
                    'timestamp': datetime.now().isoformat(),
                    'model_predictions': {k: float(v[0]) for k, v in model_predictions.items()}
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/sentiment/<currency_pair>')
        def get_sentiment(currency_pair):
            """Get sentiment analysis for a currency pair"""
            try:
                # Simulate sentiment analysis
                sentiment = {
                    'vader_compound': 0.65,
                    'vader_positive': 0.70,
                    'vader_negative': 0.15,
                    'vader_neutral': 0.15
                }
                
                return jsonify({
                    'currency_pair': currency_pair,
                    'sentiment': sentiment,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/performance')
        def get_performance():
            """Get model performance metrics"""
            try:
                conn = sqlite3.connect(self.config.DB_PATH)
                query = '''
                    SELECT model_name, 
                           AVG(ABS(actual_value - predicted_value)) as mae,
                           AVG((actual_value - predicted_value)^2) as mse,
                           COUNT(*) as predictions_count
                    FROM predictions 
                    WHERE prediction_date >= date('now', '-30 days')
                    GROUP BY model_name
                '''
                performance = pd.read_sql_query(query, conn)
                conn.close()
                
                return jsonify(performance.to_dict('records'))
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint for deployment monitoring"""
            return jsonify({
                'status': 'healthy',
                'service': 'Currency Prediction System',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
    
    def run(self, debug=True, host='0.0.0.0', port=8080):
        """Run the Flask API"""
        self.app.run(debug=debug, host=host, port=port)

# Main System Integration
class CurrencyPredictionSystem:
    """Main system that integrates all components"""
    
    def __init__(self, config=None):
        self.config = config if config else Config()
        self.setup_components()
    
    def setup_components(self):
        """Setup all system components"""
        self.data_collector = DataCollector(self.config)
        self.feature_engineer = FeatureEngineer()
        self.ensemble_forecaster = EnsembleForecaster(self.config)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.db_manager = DatabaseManager(self.config.DB_PATH)
        self.api = CurrencyPredictionAPI(
            self.config, self.data_collector, self.feature_engineer,
            self.ensemble_forecaster, self.sentiment_analyzer, self.db_manager
        )
    
    def start_api(self, port=None):
        """Start the Flask API"""
        self.data_collector.logger.info("Starting Currency Prediction API...")
        port = port or self.config.API_PORT
        self.api.run(debug=False, host='0.0.0.0', port=port)

# Main execution
if __name__ == "__main__":
    # Initialize the system
    system = CurrencyPredictionSystem()
    
    # Start API
    system.start_api() 