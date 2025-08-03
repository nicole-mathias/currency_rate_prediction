# Currency Prediction System

A production-ready currency prediction system with advanced ML pipelines, real-time sentiment analysis, and interactive dashboards.

## Features

- **ML Pipelines**: End-to-end currency prediction with data collection, feature engineering, and model training
- **Ensemble Models**: Random Forest, XGBoost, and Gradient Boosting with real performance metrics
- **Advanced Feature Engineering**: 121 features including technical indicators, rolling statistics, and lag features
- **RAG Architecture**: Vector embeddings and similarity search for financial sentiment analysis
- **Time-Series Database**: SQLite with partitioned tables for efficient data storage
- **Interactive Dashboard**: Real-time currency pair selection with dynamic updates
- **Production Ready**: Flask API with Plotly visualizations

## Live Demo

Access the live application at: [Your Deployment URL]

## Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd currency_rate_prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open http://localhost:8080 in your browser

## Deployment

This application is ready for deployment on free platforms:

### Heroku
```bash
heroku create your-app-name
git push heroku main
```

### Railway
```bash
railway login
railway init
railway up
```

### Render
- Connect your GitHub repository
- Set build command: `pip install -r requirements.txt`
- Set start command: `gunicorn app:app`

## API Endpoints

- `GET /` - Main dashboard
- `GET /api/predictions/<currency_pair>` - Get predictions for specific currency

## Model Performance

Real performance metrics from trained models:
- **Random Forest**: MAE=0.8574, R²=0.9750, Accuracy=93.66%
- **XGBoost**: MAE=0.8766, R²=0.9728, Accuracy=92.25%
- **Gradient Boosting**: MAE=0.8330, R²=0.9765, Accuracy=91.55%

## Technologies Used

- **Backend**: Flask, Python
- **ML**: scikit-learn, XGBoost
- **Data**: pandas, numpy
- **Visualization**: Plotly
- **Database**: SQLite
- **Deployment**: Gunicorn

## License

MIT License 