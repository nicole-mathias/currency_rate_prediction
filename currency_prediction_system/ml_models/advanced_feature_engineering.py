#!/usr/bin/env python3
"""
Advanced Feature Engineering with Technical Indicators
===================================================

This module implements advanced feature engineering including:
- Rolling statistics
- Technical indicators (RSI, MACD, Bollinger Bands)
- Cross-currency correlation matrices
- Fourier features for seasonality
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
# import talib  # Commented out due to installation issues
from scipy import stats
from scipy.fft import fft, ifft
import logging

class AdvancedFeatureEngineer:
    """Advanced feature engineering with technical indicators"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def create_advanced_technical_indicators(self, df):
        """Create advanced technical indicators"""
        self.logger.info("Creating advanced technical indicators...")
        
        # Ensure we have OHLCV data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            # If we don't have OHLCV, create synthetic data
            df = self._create_synthetic_ohlcv(df)
        
        # Simple RSI calculation (without TA-Lib)
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi'] = calculate_rsi(df['close'])
        
        # Simple MACD calculation
        def calculate_macd(prices, fast=12, slow=26, signal=9):
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        
        macd_line, signal_line, histogram = calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Simple Bollinger Bands calculation
        def calculate_bollinger_bands(prices, period=20, std_dev=2):
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, sma, lower_band
        
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Simple Stochastic Oscillator
        def calculate_stochastic(high, low, close, period=14):
            lowest_low = low.rolling(window=period).min()
            highest_high = high.rolling(window=period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=3).mean()
            return k_percent, d_percent
        
        stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # Simple Williams %R
        def calculate_williams_r(high, low, close, period=14):
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return williams_r
        
        df['williams_r'] = calculate_williams_r(df['high'], df['low'], df['close'])
        
        # Simple ATR calculation
        def calculate_atr(high, low, close, period=14):
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr
        
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
        
        # Simple CCI calculation
        def calculate_cci(high, low, close, period=20):
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad)
            return cci
        
        df['cci'] = calculate_cci(df['high'], df['low'], df['close'])
        
        # Simple MFI calculation
        def calculate_mfi(high, low, close, volume, period=14):
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            
            positive_mf = positive_flow.rolling(window=period).sum()
            negative_mf = negative_flow.rolling(window=period).sum()
            
            mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
            return mfi
        
        df['mfi'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
        
        # Simple OBV calculation
        def calculate_obv(close, volume):
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
        
        df['obv'] = calculate_obv(df['close'], df['volume'])
        
        return df
    
    def create_rolling_statistics(self, df, windows=[5, 10, 20, 50]):
        """Create advanced rolling statistics"""
        self.logger.info("Creating rolling statistics...")
        
        for window in windows:
            # Price-based rolling statistics
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'std_{window}'] = df['close'].rolling(window=window).std()
            df[f'min_{window}'] = df['close'].rolling(window=window).min()
            df[f'max_{window}'] = df['close'].rolling(window=window).max()
            df[f'range_{window}'] = df[f'max_{window}'] - df[f'min_{window}']
            
            # Volume-based rolling statistics
            if 'volume' in df.columns:
                df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
                df[f'volume_std_{window}'] = df['volume'].rolling(window=window).std()
            
            # Volatility measures
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window=window).std()
            df[f'log_returns_{window}'] = np.log(df['close'] / df['close'].shift(1)).rolling(window=window).mean()
        
        return df
    
    def create_cross_currency_correlations(self, currency_data):
        """Create cross-currency correlation matrices"""
        self.logger.info("Creating cross-currency correlations...")
        
        # Calculate returns for all currencies
        returns_data = {}
        for currency, data in currency_data.items():
            if 'close' in data.columns:
                returns_data[currency] = data['close'].pct_change()
        
        # Create correlation matrix
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        # Create lagged correlations
        lagged_correlations = {}
        for lag in [1, 2, 3, 5, 10]:
            lagged_corr = returns_df.corr(returns_df.shift(lag))
            lagged_correlations[f'lag_{lag}'] = lagged_corr
        
        return correlation_matrix, lagged_correlations
    
    def create_fourier_features(self, df, periods=[7, 30, 90]):
        """Create Fourier features for seasonality detection"""
        self.logger.info("Creating Fourier features...")
        
        # Get the time series data
        ts = df['close'].values
        
        for period in periods:
            # Apply FFT
            fft_vals = fft(ts)
            fft_norm = np.abs(fft_vals)
            
            # Extract dominant frequencies
            dominant_freq_idx = np.argsort(fft_norm)[-5:]  # Top 5 frequencies
            
            for i, freq_idx in enumerate(dominant_freq_idx):
                # Create sinusoidal features
                freq = freq_idx / len(ts)
                df[f'fourier_sin_{period}_{i}'] = np.sin(2 * np.pi * freq * np.arange(len(ts)))
                df[f'fourier_cos_{period}_{i}'] = np.cos(2 * np.pi * freq * np.arange(len(ts)))
        
        return df
    
    def create_lag_features(self, df, lags=[1, 2, 3, 5, 10, 20]):
        """Create lagged features"""
        self.logger.info("Creating lag features...")
        
        for lag in lags:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between indicators"""
        self.logger.info("Creating interaction features...")
        
        # Price-volume interactions
        if 'volume' in df.columns:
            df['price_volume'] = df['close'] * df['volume']
            df['price_volume_ratio'] = df['close'] / df['volume']
        
        # Technical indicator interactions
        if 'rsi' in df.columns and 'macd' in df.columns:
            df['rsi_macd_interaction'] = df['rsi'] * df['macd']
        
        if 'bb_position' in df.columns and 'rsi' in df.columns:
            df['bb_rsi_interaction'] = df['bb_position'] * df['rsi']
        
        return df
    
    def create_statistical_features(self, df, windows=[20, 50]):
        """Create statistical features"""
        self.logger.info("Creating statistical features...")
        
        for window in windows:
            # Skewness and kurtosis
            df[f'skewness_{window}'] = df['close'].rolling(window=window).skew()
            df[f'kurtosis_{window}'] = df['close'].rolling(window=window).kurt()
            
            # Percentile features
            df[f'p25_{window}'] = df['close'].rolling(window=window).quantile(0.25)
            df[f'p75_{window}'] = df['close'].rolling(window=window).quantile(0.75)
            df[f'iqr_{window}'] = df[f'p75_{window}'] - df[f'p25_{window}']
        
        return df
    
    def _create_synthetic_ohlcv(self, df):
        """Create synthetic OHLCV data if not available"""
        if 'close' in df.columns:
            # Create synthetic OHLCV from close price
            df['open'] = df['close'] * (1 + np.random.normal(0, 0.001, len(df)))
            df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.002, len(df)))
            df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.002, len(df)))
            df['volume'] = np.random.uniform(1000, 10000, len(df))
        
        return df
    
    def engineer_all_features(self, df, currency_data=None):
        """Engineer all advanced features"""
        self.logger.info("Engineering all advanced features...")
        
        # Create all technical indicators
        df = self.create_advanced_technical_indicators(df)
        
        # Create rolling statistics
        df = self.create_rolling_statistics(df)
        
        # Create lag features
        df = self.create_lag_features(df)
        
        # Create Fourier features
        df = self.create_fourier_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Create statistical features
        df = self.create_statistical_features(df)
        
        # Create cross-currency correlations if multiple currencies available
        if currency_data and len(currency_data) > 1:
            correlation_matrix, lagged_correlations = self.create_cross_currency_correlations(currency_data)
            # Store correlations for later use
            df.attrs['correlation_matrix'] = correlation_matrix
            df.attrs['lagged_correlations'] = lagged_correlations
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        self.logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
        
        return df 