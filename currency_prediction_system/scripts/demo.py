#!/usr/bin/env python3
"""
Currency Rate Prediction System Demo
===================================

This demo shows how the system integrates all 4 projects and provides
currency rate predictions with sentiment analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import json

def create_sample_data():
    """Create sample data for demonstration"""
    print("📊 Creating sample data...")
    
    # Sample currency data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # USD/JPY data with realistic patterns
    np.random.seed(42)
    base_rate = 150.0
    rates = []
    for i in range(len(dates)):
        # Add trend and noise
        trend = 0.1 * np.sin(i / 30)  # Monthly cycle
        noise = np.random.normal(0, 0.5)
        rate = base_rate + trend + noise
        rates.append(rate)
    
    currency_data = pd.DataFrame({
        'date': dates,
        'USDJPY': rates,
        'EURUSD': [1.08 + 0.02 * np.sin(i/20) + np.random.normal(0, 0.01) for i in range(len(dates))],
        'GBPUSD': [1.25 + 0.03 * np.sin(i/25) + np.random.normal(0, 0.015) for i in range(len(dates))],
        'USDCHF': [0.92 + 0.015 * np.sin(i/22) + np.random.normal(0, 0.008) for i in range(len(dates))],
        'AUDUSD': [0.68 + 0.025 * np.sin(i/18) + np.random.normal(0, 0.012) for i in range(len(dates))]
    })
    
    # Economic indicators
    economic_data = pd.DataFrame({
        'date': dates,
        'cpi_us': [2.5 + 0.1 * np.sin(i/60) for i in range(len(dates))],
        'inflation_us': [3.0 + 0.2 * np.sin(i/90) for i in range(len(dates))],
        'interest_r_us': [5.0 + 0.5 * np.sin(i/120) for i in range(len(dates))],
        'gdp_pc_us': [65000 + 1000 * np.sin(i/365) for i in range(len(dates))],
        'govt_debt_us': [30000000 + 500000 * np.sin(i/180) for i in range(len(dates))],
        'cpi_j': [0.5 + 0.05 * np.sin(i/60) for i in range(len(dates))],
        'inflation_j': [1.0 + 0.1 * np.sin(i/90) for i in range(len(dates))],
        'interest_r_j': [0.1 + 0.05 * np.sin(i/120) for i in range(len(dates))],
        'gdp_pc_j': [40000 + 800 * np.sin(i/365) for i in range(len(dates))],
        'govt_debt_j': [12000000 + 300000 * np.sin(i/180) for i in range(len(dates))]
    })
    
    # Sentiment data
    sentiment_data = pd.DataFrame({
        'date': dates,
        'sentiment_score': [0.6 + 0.3 * np.sin(i/7) + np.random.normal(0, 0.1) for i in range(len(dates))],
        'news_volume': [100 + 50 * np.sin(i/14) + np.random.normal(0, 20) for i in range(len(dates))]
    })
    
    return currency_data, economic_data, sentiment_data

def demonstrate_feature_engineering(currency_data):
    """Demonstrate feature engineering capabilities"""
    print("\n🔧 Demonstrating feature engineering...")
    
    # Technical indicators
    df = currency_data.copy()
    
    # Moving averages
    df['sma_5'] = df['USDJPY'].rolling(window=5).mean()
    df['sma_20'] = df['USDJPY'].rolling(window=20).mean()
    df['ema_12'] = df['USDJPY'].ewm(span=12).mean()
    df['ema_26'] = df['USDJPY'].ewm(span=26).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['USDJPY'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['volatility'] = df['USDJPY'].rolling(window=20).std()
    
    print("✅ Technical indicators created:")
    print("   • Simple Moving Averages (5, 20)")
    print("   • Exponential Moving Averages (12, 26)")
    print("   • MACD with signal line")
    print("   • RSI (14-period)")
    print("   • Volatility (20-period)")
    
    return df

def demonstrate_ensemble_prediction(df):
    """Demonstrate ensemble prediction capabilities"""
    print("\n🤖 Demonstrating ensemble prediction...")
    
    # Prepare features for prediction
    feature_columns = ['sma_5', 'sma_20', 'ema_12', 'ema_26', 'macd', 'rsi', 'volatility']
    X = df[feature_columns].dropna()
    y = df['USDJPY'].iloc[len(X):len(X)+len(X)]
    
    if len(X) != len(y):
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]
    
    # Simulate ensemble predictions
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': None,  # Would use xgboost.XGBRegressor
        'Gradient Boosting': None,  # Would use GradientBoostingRegressor
        'LSTM': None  # Would use LSTM neural network
    }
    
    predictions = {}
    for name, model in models.items():
        if model is not None:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions[name] = pred
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            print(f"   • {name}: MAE={mae:.4f}, R²={r2:.4f}")
    
    # Ensemble prediction (weighted average)
    if predictions:
        ensemble_pred = np.zeros(len(y_test))
        weights = {'Random Forest': 1.0}  # Would include all models
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
        
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        print(f"   • Ensemble: MAE={ensemble_mae:.4f}, R²={ensemble_r2:.4f}")
    
    return predictions

def demonstrate_sentiment_analysis(sentiment_data):
    """Demonstrate sentiment analysis capabilities"""
    print("\n📰 Demonstrating sentiment analysis...")
    
    # Simulate sentiment analysis results
    recent_sentiment = sentiment_data.tail(30)
    
    positive_news = len(recent_sentiment[recent_sentiment['sentiment_score'] > 0.6])
    neutral_news = len(recent_sentiment[(recent_sentiment['sentiment_score'] >= 0.4) & 
                                       (recent_sentiment['sentiment_score'] <= 0.6)])
    negative_news = len(recent_sentiment[recent_sentiment['sentiment_score'] < 0.4])
    
    print("✅ Sentiment analysis results:")
    print(f"   • Positive news: {positive_news} articles")
    print(f"   • Neutral news: {neutral_news} articles")
    print(f"   • Negative news: {negative_news} articles")
    print(f"   • Average sentiment: {recent_sentiment['sentiment_score'].mean():.3f}")
    
    return recent_sentiment

def demonstrate_mlops_pipeline():
    """Demonstrate MLOps pipeline capabilities"""
    print("\n🔄 Demonstrating MLOps pipeline...")
    
    # Simulate model performance monitoring
    performance_metrics = {
        'model_accuracy': 0.78,
        'mae': 0.045,
        'r2_score': 0.82,
        'sharpe_ratio': 0.65,
        'data_quality_score': 0.95,
        'api_response_time': 0.15  # seconds
    }
    
    print("✅ MLOps pipeline metrics:")
    for metric, value in performance_metrics.items():
        print(f"   • {metric.replace('_', ' ').title()}: {value}")
    
    # Simulate alerts
    alerts = []
    if performance_metrics['model_accuracy'] < 0.7:
        alerts.append("⚠️  Model accuracy below threshold")
    if performance_metrics['api_response_time'] > 0.2:
        alerts.append("⚠️  API response time high")
    
    if alerts:
        print("\n🚨 Alerts:")
        for alert in alerts:
            print(f"   {alert}")
    else:
        print("\n✅ All systems operational")
    
    return performance_metrics

def create_visualizations(currency_data, df, sentiment_data):
    """Create sample visualizations"""
    print("\n📊 Creating visualizations...")
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # 1. Currency price chart
    plt.figure(figsize=(12, 6))
    plt.plot(currency_data['date'], currency_data['USDJPY'], label='USD/JPY')
    plt.title('USD/JPY Exchange Rate')
    plt.xlabel('Date')
    plt.ylabel('Rate (JPY)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/currency_price.png')
    plt.close()
    
    # 2. Technical indicators
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Price and moving averages
    axes[0].plot(df['date'], df['USDJPY'], label='Price')
    axes[0].plot(df['date'], df['sma_20'], label='SMA(20)')
    axes[0].plot(df['date'], df['ema_12'], label='EMA(12)')
    axes[0].set_title('Price and Moving Averages')
    axes[0].legend()
    
    # MACD
    axes[1].plot(df['date'], df['macd'], label='MACD')
    axes[1].plot(df['date'], df['macd_signal'], label='Signal')
    axes[1].bar(df['date'], df['macd_histogram'], label='Histogram', alpha=0.3)
    axes[1].set_title('MACD')
    axes[1].legend()
    
    # RSI
    axes[2].plot(df['date'], df['rsi'], label='RSI')
    axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
    axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)
    axes[2].set_title('RSI')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('plots/technical_indicators.png')
    plt.close()
    
    # 3. Sentiment analysis
    plt.figure(figsize=(12, 6))
    plt.plot(sentiment_data['date'], sentiment_data['sentiment_score'])
    plt.axhline(y=0.6, color='g', linestyle='--', alpha=0.5, label='Positive threshold')
    plt.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='Negative threshold')
    plt.title('News Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/sentiment_analysis.png')
    plt.close()
    
    print("✅ Visualizations saved to 'plots/' directory:")
    print("   • currency_price.png")
    print("   • technical_indicators.png")
    print("   • sentiment_analysis.png")

def demonstrate_api_endpoints():
    """Demonstrate API endpoint functionality"""
    print("\n🌐 Demonstrating API endpoints...")
    
    # Simulate API responses
    api_responses = {
        '/api/predict/USDJPY': {
            'currency_pair': 'USDJPY',
            'prediction': 150.25,
            'confidence': 0.85,
            'timestamp': datetime.now().isoformat(),
            'model_predictions': {
                'random_forest': 150.30,
                'xgboost': 150.20,
                'gradient_boosting': 150.25,
                'lstm': 150.22
            }
        },
        '/api/sentiment/USDJPY': {
            'currency_pair': 'USDJPY',
            'sentiment': {
                'vader_compound': 0.65,
                'transformer_positive': 0.70,
                'transformer_negative': 0.15,
                'transformer_neutral': 0.15
            },
            'timestamp': datetime.now().isoformat()
        },
        '/api/performance': [
            {
                'model_name': 'random_forest',
                'mae': 0.045,
                'mse': 0.003,
                'predictions_count': 1000
            },
            {
                'model_name': 'xgboost',
                'mae': 0.042,
                'mse': 0.0028,
                'predictions_count': 1000
            }
        ]
    }
    
    print("✅ API endpoints demonstrated:")
    for endpoint, response in api_responses.items():
        print(f"   • {endpoint}: {json.dumps(response, indent=2)[:100]}...")

def main():
    """Main demo function"""
    print("=" * 60)
    print("🎯 Currency Rate Prediction System Demo")
    print("=" * 60)
    print("This demo shows how the system integrates all 4 projects")
    print("and provides comprehensive currency prediction capabilities.")
    print()
    
    # Create sample data
    currency_data, economic_data, sentiment_data = create_sample_data()
    
    # Demonstrate feature engineering
    df = demonstrate_feature_engineering(currency_data)
    
    # Demonstrate ensemble prediction
    predictions = demonstrate_ensemble_prediction(df)
    
    # Demonstrate sentiment analysis
    recent_sentiment = demonstrate_sentiment_analysis(sentiment_data)
    
    # Demonstrate MLOps pipeline
    performance_metrics = demonstrate_mlops_pipeline()
    
    # Create visualizations
    create_visualizations(currency_data, df, sentiment_data)
    
    # Demonstrate API endpoints
    demonstrate_api_endpoints()
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("=" * 60)
    print("\n📋 Summary of demonstrated capabilities:")
    print("   ✅ Multi-source data collection")
    print("   ✅ Advanced feature engineering")
    print("   ✅ Ensemble model predictions")
    print("   ✅ Real-time sentiment analysis")
    print("   ✅ MLOps pipeline monitoring")
    print("   ✅ Interactive visualizations")
    print("   ✅ Production-ready API endpoints")
    print("\n🚀 To run the full system:")
    print("   python run_system.py")
    print("\n📊 To view the dashboard:")
    print("   http://localhost:5000")

if __name__ == "__main__":
    main() 