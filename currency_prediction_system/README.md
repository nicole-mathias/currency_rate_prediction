# Currency Rate Prediction System

A comprehensive machine learning system for currency rate prediction that integrates data collection, processing, modeling, and deployment. This system provides real-time currency predictions using ensemble machine learning models and sentiment analysis.

## Project Structure

```
currency_prediction_system/
├── config/                     # Configuration files
│   └── config.py              # Main configuration settings
├── data_collection/           # Data collection modules
│   ├── fred_data.py          # FRED API data collection
│   ├── scraping.py           # Web scraping utilities
│   └── datasets/             # Historical datasets
├── data_processing/          # Data processing and integration
│   ├── data_integration.py   # Unified data integration
│   ├── binning/              # Data binning algorithms
│   └── clustering/           # Clustering algorithms
├── ml_models/               # Machine learning models
│   ├── classifiers.py        # Classification models
│   ├── regression_models.py  # Regression models
│   ├── clustering/           # Clustering implementations
│   └── sentiment_analysis.py # NLP and sentiment analysis
├── api_dashboard/           # Web API and dashboard
│   ├── currency_prediction_system_simple.py
│   └── templates/           # HTML templates
├── mlops/                   # MLOps and monitoring
├── scripts/                 # Utility scripts
│   ├── main.py             # Main launcher
│   └── demo.py             # Demo script
├── logs/                   # Log files
├── models/                 # Trained model files
├── plots/                  # Generated visualizations
└── requirements.txt        # Python dependencies
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the system:**
   ```bash
   python main.py
   ```

3. **Access the API:**
   - API: http://localhost:8080
   - Dashboard: http://localhost:8080

## System Components

### 1. Data Collection (`data_collection/`)
- **FRED API Integration**: Economic indicators from Federal Reserve
- **Web Scraping**: News and market data collection
- **Yahoo Finance**: Stock market data
- **Multi-source Integration**: Unified data pipeline

### 2. Data Processing (`data_processing/`)
- **Data Integration**: Combines data from all sources
- **Feature Engineering**: Technical indicators and rolling statistics
- **Data Cleaning**: Missing value handling and normalization
- **Binning & Clustering**: Data preprocessing algorithms

### 3. Machine Learning Models (`ml_models/`)
- **Classification Models**: 
  - Logistic Regression
  - K-Nearest Neighbors
  - Naive Bayes
  - Support Vector Machine
  - Random Forest
  - Decision Trees
- **Regression Models**:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest Regressor
  - Gradient Boosting
- **Clustering Models**:
  - K-Means
  - DBSCAN
  - Gaussian Mixture Models
- **Sentiment Analysis**:
  - VADER Sentiment
  - AFINN Lexicon
  - Transformer-based NLP

### 4. API & Dashboard (`api_dashboard/`)
- **Flask API**: RESTful endpoints for predictions
- **Interactive Dashboard**: Plotly-based visualizations
- **Real-time Updates**: Live data streaming
- **Model Performance**: Monitoring and metrics

### 5. MLOps (`mlops/`)
- **Model Versioning**: Track model iterations
- **Performance Monitoring**: Real-time model evaluation
- **Automated Retraining**: Scheduled model updates
- **Backtesting**: Historical performance validation

## Configuration

Edit `config/config.py` to customize:
- API keys and endpoints
- Currency pairs to track
- Model parameters
- Database settings
- Logging configuration

## API Endpoints

### Predictions
```bash
# Get currency prediction
curl http://localhost:8080/api/predict/USDJPY

# Response:
{
  "currency_pair": "USDJPY",
  "prediction": 150.25,
  "confidence": 0.85,
  "timestamp": "2024-01-01T12:00:00",
  "model_predictions": {
    "random_forest": 150.30,
    "xgboost": 150.20,
    "gradient_boosting": 150.25
  }
}
```

### Sentiment Analysis
```bash
# Get sentiment for currency pair
curl http://localhost:8080/api/sentiment/USDJPY

# Response:
{
  "currency_pair": "USDJPY",
  "sentiment": {
    "vader_compound": 0.65,
    "vader_positive": 0.70,
    "vader_negative": 0.15,
    "vader_neutral": 0.15
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### Performance Metrics
```bash
# Get model performance
curl http://localhost:8080/api/performance

# Response:
[
  {
    "model_name": "random_forest",
    "mae": 0.045,
    "mse": 0.003,
    "predictions_count": 1000
  }
]
```

## Key Features

### Advanced Feature Engineering
- **Technical Indicators**: SMA, EMA, MACD, RSI, Bollinger Bands
- **Rolling Statistics**: Moving averages, volatility, momentum
- **Cross-currency Correlation**: Multi-pair analysis
- **Fourier Features**: Time series decomposition

### Ensemble Forecasting
- **Multiple Models**: Random Forest, XGBoost, Gradient Boosting
- **Weighted Averaging**: Optimal model combination
- **Confidence Intervals**: Prediction uncertainty
- **Model Selection**: Dynamic model weighting

### Real-time Sentiment Analysis
- **News Processing**: Automated news collection
- **NLP Pipeline**: Text preprocessing and analysis
- **Sentiment Scoring**: Multi-lexicon approach
- **Market Impact**: Sentiment-market correlation

### Production-Ready API
- **RESTful Design**: Standard HTTP endpoints
- **Error Handling**: Comprehensive error management
- **Rate Limiting**: API usage controls
- **Documentation**: Auto-generated API docs

## Visualization Features

- **Interactive Charts**: Plotly-based dashboards
- **Technical Analysis**: Real-time indicator plots
- **Performance Metrics**: Model evaluation charts
- **Sentiment Trends**: News sentiment visualization
- **Correlation Matrix**: Multi-currency relationships

## MLOps Pipeline

### Model Lifecycle
1. **Data Collection**: Automated data gathering
2. **Feature Engineering**: Real-time feature creation
3. **Model Training**: Automated model training
4. **Performance Evaluation**: Continuous monitoring
5. **Model Deployment**: Seamless model updates

### Monitoring
- **Model Accuracy**: Real-time performance tracking
- **Data Quality**: Automated data validation
- **API Performance**: Response time monitoring
- **System Health**: Comprehensive logging

## Development

### Running Individual Components

```bash
# Data collection only
python data_collection/fred_data.py

# Data processing only
python data_processing/data_integration.py

# ML models only
python ml_models/classifiers.py

# API only
python api_dashboard/currency_prediction_system_simple.py
```

### Adding New Models

1. Create model file in `ml_models/`
2. Implement standard interface
3. Add to ensemble in main system
4. Update configuration

### Extending Data Sources

1. Add collection script to `data_collection/`
2. Update integration in `data_processing/`
3. Modify configuration
4. Test with main system

## Logging

Logs are stored in `logs/` directory:
- `system_launcher.log`: Main system logs
- `currency_prediction.log`: API logs
- `data_integration.log`: Data processing logs



## Acknowledgments

- Federal Reserve Economic Data (FRED)
- Yahoo Finance API
