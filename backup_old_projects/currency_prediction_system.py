#!/usr/bin/env python3
"""
Currency Rate Prediction System
==============================

A comprehensive ML pipeline for currency rate prediction that integrates:
- Automated data collection from multiple sources
- Advanced feature engineering and time-series analysis
- Ensemble forecasting models
- Real-time sentiment analysis
- Production-ready Flask API with interactive dashboards
- MLOps pipeline with model retraining and monitoring

This system connects all 4 data analysis projects into a unified currency prediction platform.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Data Collection
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
import pyfredapi as pf

# NLP and Sentiment Analysis
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

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

# Configuration
class Config:
    """Configuration settings for the currency prediction system"""
    
    # API Keys (should be stored in environment variables)
    FRED_API_KEY = "bb77f6daf770fdf4461dd2500084ab11"
    ALPACA_API_KEY = "your_alpaca_key"
    ALPACA_SECRET_KEY = "your_alpaca_secret"
    
    # Currency pairs to track
    CURRENCY_PAIRS = [
        'USDJPY', 'EURUSD', 'GBPUSD', 'USDCHF', 'AUDUSD'
    ]
    
    # Economic indicators
    ECONOMIC_INDICATORS = {
        'USD': ['CPIAUCSL', 'DFF', 'A939RX0Q048SBEA', 'GFDEGDQ188S'],
        'JPY': ['DEXJPUS', 'JPNCPIALLMINMEI', 'INTDSRJPM193N', 'FPCPITOTLZGJPN'],
        'EUR': ['DEXUSEU', 'CP0000EZ17M086NEST', 'FMACBSIDX'],
        'GBP': ['DEXUSUK', 'CP0000GB17M086NEST', 'GBRCPIALLMINMEI'],
        'CHF': ['DEXSZUS', 'CHECPIALLMINMEI', 'INTDSRCHM193N'],
        'AUD': ['DEXUSAL', 'AUSCPIALLMINMEI', 'INTDSRAUM193N']
    }
    
    # Model parameters
    LSTM_UNITS = 50
    DROPOUT_RATE = 0.2
    EPOCHS = 100
    BATCH_SIZE = 32
    
    # Database
    DB_PATH = 'currency_prediction.db'
    
    # Logging
    LOG_LEVEL = logging.INFO
    LOG_FILE = 'currency_prediction.log'

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
    
    def collect_news_data(self, keywords, days_back=30):
        """Collect news data for sentiment analysis"""
        # This would integrate with news APIs like NewsAPI, GNews, etc.
        self.logger.info(f"Collecting news data for keywords: {keywords}")
        
        # Placeholder for news collection
        # In production, this would use actual news APIs
        news_data = pd.DataFrame({
            'date': pd.date_range(start=datetime.now() - timedelta(days=days_back), 
                                end=datetime.now(), freq='D'),
            'title': ['Sample news title'] * days_back,
            'content': ['Sample news content'] * days_back,
            'source': ['Reuters'] * days_back
        })
        
        return news_data

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
    
    def create_cross_currency_features(self, currency_data):
        """Create cross-currency correlation features"""
        # This would analyze correlations between different currency pairs
        correlations = currency_data.corr()
        
        # Add correlation-based features
        for pair1 in currency_data.columns:
            for pair2 in currency_data.columns:
                if pair1 != pair2:
                    currency_data[f'corr_{pair1}_{pair2}'] = correlations.loc[pair1, pair2]
        
        return currency_data
    
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

# Ensemble Model Module
class EnsembleForecaster:
    """Ensemble forecasting models combining multiple algorithms"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scaler = StandardScaler()
        
    def prepare_lstm_data(self, data, lookback=60):
        """Prepare data for LSTM model"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM neural network"""
        model = Sequential([
            LSTM(units=self.config.LSTM_UNITS, return_sequences=True, input_shape=input_shape),
            Dropout(self.config.DROPOUT_RATE),
            LSTM(units=self.config.LSTM_UNITS, return_sequences=True),
            Dropout(self.config.DROPOUT_RATE),
            LSTM(units=self.config.LSTM_UNITS),
            Dropout(self.config.DROPOUT_RATE),
            Dense(units=1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
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
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        # 3. Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb_model
        
        # 4. LSTM (if time series data)
        if len(X_train.shape) == 3:  # 3D data for LSTM
            lstm_model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
            lstm_model.fit(X_train, y_train, epochs=self.config.EPOCHS, 
                         batch_size=self.config.BATCH_SIZE, verbose=0)
            self.models['lstm'] = lstm_model
        
        self.logger.info("Ensemble models trained successfully")
        return self.models
    
    def ensemble_predict(self, X):
        """Make ensemble predictions"""
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'lstm':
                pred = model.predict(X)
            else:
                pred = model.predict(X)
            predictions[name] = pred
        
        # Weighted average of predictions
        weights = {
            'random_forest': 0.3,
            'xgboost': 0.3,
            'gradient_boosting': 0.2,
            'lstm': 0.2
        }
        
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            if name in weights:
                ensemble_pred += weights[name] * pred.flatten()
        
        return ensemble_pred, predictions

# Sentiment Analysis Module
class SentimentAnalyzer:
    """Real-time sentiment analysis using NLP transformers"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        # Load transformer model for advanced sentiment analysis
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        
    def analyze_sentiment(self, text):
        """Analyze sentiment using multiple methods"""
        # VADER sentiment
        vader_scores = self.sia.polarity_scores(text)
        
        # Transformer-based sentiment
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            transformer_scores = torch.softmax(outputs.logits, dim=1)
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'transformer_positive': transformer_scores[0][2].item(),
            'transformer_negative': transformer_scores[0][0].item(),
            'transformer_neutral': transformer_scores[0][1].item()
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
    """MySQL-like database management for historical data storage"""
    
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
                transformer_positive REAL,
                transformer_negative REAL,
                transformer_neutral REAL,
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
                feature_columns = [col for col in data.columns if col not in ['id', 'currency_pair', 'date', 'created_at']]
                X = data[feature_columns].dropna()
                
                if len(X) == 0:
                    return jsonify({'error': 'No valid features'}), 400
                
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
                # Get recent news and sentiment
                news_data = self.data_collector.collect_news_data([currency_pair])
                sentiments = self.sentiment_analyzer.process_news_batch(news_data)
                
                avg_sentiment = sentiments.mean().to_dict()
                
                return jsonify({
                    'currency_pair': currency_pair,
                    'sentiment': avg_sentiment,
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
    
    def run(self, debug=True, host='0.0.0.0', port=5000):
        """Run the Flask API"""
        self.app.run(debug=debug, host=host, port=port)

# MLOps Pipeline Module
class MLOpsPipeline:
    """MLOps pipeline with model retraining, monitoring, and backtesting"""
    
    def __init__(self, config, ensemble_forecaster, db_manager):
        self.config = config
        self.ensemble_forecaster = ensemble_forecaster
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    def retrain_models(self, currency_pair):
        """Retrain models with latest data"""
        self.logger.info(f"Retraining models for {currency_pair}")
        
        # Get latest data
        data = self.db_manager.get_historical_data(
            currency_pair,
            (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d'),  # 3 years
            datetime.now().strftime('%Y-%m-%d')
        )
        
        if len(data) < 365:
            self.logger.warning(f"Insufficient data for {currency_pair}")
            return False
        
        # Feature engineering
        data = self.feature_engineer.create_technical_indicators(data)
        data = self.feature_engineer.create_rolling_statistics(data)
        data = self.feature_engineer.create_lag_features(data)
        
        # Prepare features and target
        feature_columns = [col for col in data.columns if col not in ['id', 'currency_pair', 'date', 'created_at']]
        X = data[feature_columns].dropna()
        y = data['Close'].iloc[len(X):len(X)+len(X)]
        
        if len(X) != len(y):
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Retrain models
        self.ensemble_forecaster.train_ensemble_models(X_train, y_train, X_test, y_test)
        
        # Save models
        self.save_models(currency_pair)
        
        self.logger.info(f"Models retrained successfully for {currency_pair}")
        return True
    
    def save_models(self, currency_pair):
        """Save trained models"""
        model_dir = f'models/{currency_pair}'
        os.makedirs(model_dir, exist_ok=True)
        
        for name, model in self.ensemble_forecaster.models.items():
            if name == 'lstm':
                model.save(f'{model_dir}/{name}_model.h5')
            else:
                joblib.dump(model, f'{model_dir}/{name}_model.pkl')
    
    def load_models(self, currency_pair):
        """Load trained models"""
        model_dir = f'models/{currency_pair}'
        
        for name in self.ensemble_forecaster.models.keys():
            if name == 'lstm':
                from tensorflow.keras.models import load_model
                self.ensemble_forecaster.models[name] = load_model(f'{model_dir}/{name}_model.h5')
            else:
                self.ensemble_forecaster.models[name] = joblib.load(f'{model_dir}/{name}_model.pkl')
    
    def backtest_strategy(self, currency_pair, start_date, end_date):
        """Backtest prediction strategy against actual market movements"""
        self.logger.info(f"Backtesting strategy for {currency_pair}")
        
        # Get historical data
        data = self.db_manager.get_historical_data(currency_pair, start_date, end_date)
        
        # Feature engineering
        data = self.feature_engineer.create_technical_indicators(data)
        data = self.feature_engineer.create_rolling_statistics(data)
        data = self.feature_engineer.create_lag_features(data)
        
        # Prepare features
        feature_columns = [col for col in data.columns if col not in ['id', 'currency_pair', 'date', 'created_at']]
        X = data[feature_columns].dropna()
        y = data['Close'].iloc[len(X):len(X)+len(X)]
        
        if len(X) != len(y):
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]
        
        # Make predictions
        predictions, _ = self.ensemble_forecaster.ensemble_predict(X)
        
        # Calculate performance metrics
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        # Calculate trading returns
        returns = []
        for i in range(1, len(predictions)):
            if predictions[i] > predictions[i-1]:  # Predicted up
                actual_return = (y.iloc[i] - y.iloc[i-1]) / y.iloc[i-1]
                returns.append(actual_return)
            else:  # Predicted down
                actual_return = (y.iloc[i-1] - y.iloc[i]) / y.iloc[i-1]
                returns.append(actual_return)
        
        total_return = sum(returns)
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        results = {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'predictions': predictions.tolist(),
            'actual': y.tolist()
        }
        
        self.logger.info(f"Backtest completed for {currency_pair}")
        return results

# Main System Integration
class CurrencyPredictionSystem:
    """Main system that integrates all components"""
    
    def __init__(self):
        self.config = Config()
        self.setup_components()
    
    def setup_components(self):
        """Setup all system components"""
        self.data_collector = DataCollector(self.config)
        self.feature_engineer = FeatureEngineer()
        self.ensemble_forecaster = EnsembleForecaster(self.config)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.db_manager = DatabaseManager(self.config.DB_PATH)
        self.mlops_pipeline = MLOpsPipeline(self.config, self.ensemble_forecaster, self.db_manager)
        self.api = CurrencyPredictionAPI(
            self.config, self.data_collector, self.feature_engineer,
            self.ensemble_forecaster, self.sentiment_analyzer, self.db_manager
        )
    
    def run_data_collection_pipeline(self):
        """Run automated data collection pipeline"""
        self.data_collector.logger.info("Starting data collection pipeline...")
        
        # Collect FRED data
        all_indicators = []
        for currency, indicators in self.config.ECONOMIC_INDICATORS.items():
            all_indicators.extend(indicators)
        
        fred_data = self.data_collector.collect_fred_data(all_indicators)
        
        # Collect Yahoo Finance data
        yahoo_data = self.data_collector.collect_yahoo_finance_data(self.config.CURRENCY_PAIRS)
        
        # Store data in database
        for currency_pair, data in yahoo_data.items():
            self.db_manager.store_currency_data(data, currency_pair)
        
        self.data_collector.logger.info("Data collection pipeline completed")
    
    def run_training_pipeline(self):
        """Run model training pipeline"""
        self.data_collector.logger.info("Starting training pipeline...")
        
        for currency_pair in self.config.CURRENCY_PAIRS:
            self.mlops_pipeline.retrain_models(currency_pair)
        
        self.data_collector.logger.info("Training pipeline completed")
    
    def run_sentiment_pipeline(self):
        """Run sentiment analysis pipeline"""
        self.data_collector.logger.info("Starting sentiment analysis pipeline...")
        
        for currency_pair in self.config.CURRENCY_PAIRS:
            news_data = self.data_collector.collect_news_data([currency_pair])
            sentiments = self.sentiment_analyzer.process_news_batch(news_data)
            
            # Store sentiment data
            conn = sqlite3.connect(self.config.DB_PATH)
            for _, row in sentiments.iterrows():
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO sentiment_scores 
                    (date, currency_pair, vader_compound, transformer_positive, 
                     transformer_negative, transformer_neutral)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (datetime.now().strftime('%Y-%m-%d'), currency_pair,
                      row['vader_compound'], row['transformer_positive'],
                      row['transformer_negative'], row['transformer_neutral']))
            conn.commit()
            conn.close()
        
        self.data_collector.logger.info("Sentiment analysis pipeline completed")
    
    def run_backtesting(self):
        """Run backtesting for all currency pairs"""
        self.data_collector.logger.info("Starting backtesting...")
        
        results = {}
        for currency_pair in self.config.CURRENCY_PAIRS:
            results[currency_pair] = self.mlops_pipeline.backtest_strategy(
                currency_pair,
                (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
        
        return results
    
    def start_api(self):
        """Start the Flask API"""
        self.data_collector.logger.info("Starting Currency Prediction API...")
        self.api.run(debug=False, host='0.0.0.0', port=5000)

# Interactive Dashboard using Plotly Dash
def create_dashboard():
    """Create interactive Plotly Dash dashboard"""
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Currency Rate Prediction Dashboard", className="text-center mb-4"),
                html.Hr()
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Currency Pair Selection"),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='currency-dropdown',
                            options=[
                                {'label': 'USD/JPY', 'value': 'USDJPY'},
                                {'label': 'EUR/USD', 'value': 'EURUSD'},
                                {'label': 'GBP/USD', 'value': 'GBPUSD'},
                                {'label': 'USD/CHF', 'value': 'USDCHF'},
                                {'label': 'AUD/USD', 'value': 'AUDUSD'}
                            ],
                            value='USDJPY'
                        )
                    ])
                ])
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Prediction"),
                    dbc.CardBody([
                        html.H2(id='prediction-value', className="text-center"),
                        html.P(id='prediction-confidence', className="text-center text-muted")
                    ])
                ])
            ], width=8)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Price Chart"),
                    dbc.CardBody([
                        dcc.Graph(id='price-chart')
                    ])
                ])
            ])
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Performance"),
                    dbc.CardBody([
                        dcc.Graph(id='performance-chart')
                    ])
                ])
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Sentiment Analysis"),
                    dbc.CardBody([
                        dcc.Graph(id='sentiment-chart')
                    ])
                ])
            ], width=6)
        ])
    ])
    
    return app

# Main execution
if __name__ == "__main__":
    # Initialize the system
    system = CurrencyPredictionSystem()
    
    # Run data collection
    system.run_data_collection_pipeline()
    
    # Run training
    system.run_training_pipeline()
    
    # Run sentiment analysis
    system.run_sentiment_pipeline()
    
    # Run backtesting
    backtest_results = system.run_backtesting()
    print("Backtesting Results:", backtest_results)
    
    # Start API
    system.start_api() 