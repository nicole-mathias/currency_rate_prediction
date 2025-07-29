"""
Configuration settings for the Currency Rate Prediction System
"""

import os
from dotenv import load_dotenv

# Load environment variables (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If dotenv is not available, continue without it
    pass

class Config:
    """Configuration settings for the currency prediction system"""
    
    # API Keys (should be stored in environment variables)
    FRED_API_KEY = os.getenv("FRED_API_KEY", "bb77f6daf770fdf4461dd2500084ab11")
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "your_alpaca_key")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "your_alpaca_secret")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "your_news_api_key")
    
    # Currency pairs to track
    CURRENCY_PAIRS = [
        'USDJPY', 'EURUSD', 'GBPUSD', 'USDCHF', 'AUDUSD'
    ]
    
    # Economic indicators by currency
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
    LOOKBACK_DAYS = 60
    
    # Database settings
    DB_PATH = 'currency_prediction.db'
    
    # Logging settings
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'currency_prediction.log'
    
    # API settings
    API_HOST = '0.0.0.0'
    API_PORT = 8080
    API_DEBUG = False
    
    # Dashboard settings
    DASHBOARD_HOST = '0.0.0.0'
    DASHBOARD_PORT = 8050
    
    # Data collection settings
    DATA_COLLECTION_INTERVAL = 3600  # seconds
    SENTIMENT_ANALYSIS_INTERVAL = 1800  # seconds
    MODEL_RETRAINING_INTERVAL = 86400  # seconds (24 hours)
    
    # Feature engineering settings
    TECHNICAL_INDICATORS = {
        'sma_periods': [5, 10, 20, 50],
        'ema_periods': [12, 26],
        'rsi_period': 14,
        'bb_period': 20,
        'bb_std': 2,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }
    
    # Ensemble model weights
    MODEL_WEIGHTS = {
        'random_forest': 0.3,
        'xgboost': 0.3,
        'gradient_boosting': 0.2,
        'lstm': 0.2
    }
    
    # Performance thresholds
    PERFORMANCE_THRESHOLDS = {
        'min_accuracy': 0.6,
        'max_mae': 0.05,
        'min_sharpe_ratio': 0.5
    }
    
    # File paths
    MODELS_DIR = 'models'
    PLOTS_DIR = 'plots'
    DATA_DIR = 'data'
    LOGS_DIR = 'logs'
    
    # Create directories if they don't exist
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [cls.MODELS_DIR, cls.PLOTS_DIR, cls.DATA_DIR, cls.LOGS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True) 