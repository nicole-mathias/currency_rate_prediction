# Currency Prediction System - Deployment Guide

## Free Deployment Options

### 1. Railway (Recommended)
- **URL**: https://railway.app
- **Steps**:
  1. Connect your GitHub repository
  2. Railway will auto-detect Python app
  3. Deploy automatically

### 2. Render
- **URL**: https://render.com
- **Steps**:
  1. Create new Web Service
  2. Connect GitHub repository
  3. Build Command: `pip install -r requirements.txt`
  4. Start Command: `gunicorn app:app`

### 3. Heroku
- **URL**: https://heroku.com
- **Steps**:
  1. Install Heroku CLI
  2. `heroku create your-app-name`
  3. `git push heroku main`

## Files Included for Deployment

- ✅ `app.py` - Main Flask application
- ✅ `requirements.txt` - Python dependencies
- ✅ `Procfile` - Process definition for gunicorn
- ✅ `runtime.txt` - Python version specification
- ✅ `real_comprehensive_data_fixed.json` - Market data

## Quick Deploy Commands

```bash
# For Railway
railway login
railway init
railway up

# For Render
# Use web interface at render.com

# For Heroku
heroku create your-app-name
git push heroku main
```

## System Features

- 📊 **5 Currency Pairs**: USD/JPY, EUR/USD, GBP/USD, USD/CHF, AUD/USD
- 🤖 **3 ML Models**: Random Forest, XGBoost, Gradient Boosting
- 📈 **Performance Metrics**: MAE, R², Directional Accuracy
- 📰 **News Sentiment**: 57,249 articles analyzed
- 🔍 **RAG System**: Vector embeddings for document retrieval
- 🎯 **Interactive Dashboard**: Real-time charts and predictions

## Data Sources

- **Yahoo Finance**: Market data for all currency pairs
- **Federal Reserve**: Economic indicators
- **News APIs**: Sentiment analysis from financial news

## Model Performance

| Currency | Best Model | MAE | R² | Accuracy |
|----------|------------|-----|----|----------|
| USD/JPY | Gradient Boosting | 1.53 | 0.48 | 84.14% |
| EUR/USD | Gradient Boosting | 0.010 | 0.79 | 81.99% |
| GBP/USD | Gradient Boosting | 0.007 | 0.81 | 81.18% |
| USD/CHF | Gradient Boosting | 0.007 | 0.83 | 79.03% |
| AUD/USD | Gradient Boosting | 0.002 | 0.92 | 84.68% | 