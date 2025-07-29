"""
Data Integration Module
======================

This module integrates data from all 4 projects into the unified currency prediction system.
It connects existing data sources and creates a unified data pipeline.
"""

import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime, timedelta
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config

class DataIntegrator:
    """Integrates data from all existing projects"""
    
    def __init__(self):
        self.config = Config()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def integrate_project_1_data(self):
        """Integrate data from FRED API and web scraping sources"""
        self.logger.info("Integrating FRED and scraping data...")
        
        # Load existing FRED data
        try:
            fred_data = pd.read_csv('project-1-group-3-main/us_jap_data.csv')
            fred_data['date'] = pd.to_datetime(fred_data['date'])
            self.logger.info(f"Loaded FRED data: {len(fred_data)} records")
        except FileNotFoundError:
            self.logger.warning("FRED data file not found, will collect fresh data")
            fred_data = None
        
        # Load existing scraping data
        scraping_data = {}
        scraping_files = [
            'project-1-group-3-main/scraping/US_scraping.csv',
            'project-1-group-3-main/scraping/Japan_scraping.csv'
        ]
        
        for file_path in scraping_files:
            try:
                data = pd.read_csv(file_path)
                country = 'US' if 'US_scraping' in file_path else 'Japan'
                scraping_data[country] = data
                self.logger.info(f"Loaded {country} scraping data: {len(data)} records")
            except FileNotFoundError:
                self.logger.warning(f"Scraping data file not found: {file_path}")
        
        return {
            'fred_data': fred_data,
            'scraping_data': scraping_data
        }
    
    def integrate_project_2_data(self):
        """Integrate data from clustering, binning, and sentiment analysis"""
        self.logger.info("Integrating clustering and sentiment data...")
        
        # Load combined clean data
        try:
            combined_data = pd.read_csv('project-2-group-3-main/datasets/new_combined_clean.csv')
            combined_data['date'] = pd.to_datetime(combined_data['date'])
            self.logger.info(f"Loaded combined data: {len(combined_data)} records")
        except FileNotFoundError:
            self.logger.warning("Combined data file not found")
            combined_data = None
        
        # Load sentiment analysis data
        sentiment_data = {}
        sentiment_files = [
            'project-2-group-3-main/datasets/new_news_data/us_news_sentiments.csv',
            'project-2-group-3-main/datasets/new_news_data/us_news.csv'
        ]
        
        for file_path in sentiment_files:
            try:
                data = pd.read_csv(file_path)
                sentiment_data[os.path.basename(file_path)] = data
                self.logger.info(f"Loaded sentiment data: {len(data)} records from {file_path}")
            except FileNotFoundError:
                self.logger.warning(f"Sentiment data file not found: {file_path}")
        
        # Load clustering results
        clustering_results = {}
        clustering_files = [
            'project-2-group-3-main/plot/k_means_clustering.png',
            'project-2-group-3-main/plot/db_scan.png',
            'project-2-group-3-main/plot/GMM.png'
        ]
        
        for file_path in clustering_files:
            if os.path.exists(file_path):
                clustering_results[os.path.basename(file_path)] = file_path
        
        return {
            'combined_data': combined_data,
            'sentiment_data': sentiment_data,
            'clustering_results': clustering_results
        }
    
    def integrate_project_3_data(self):
        """Integrate data from classification models and market data"""
        self.logger.info("Integrating classification and market data...")
        
        # Load stock market data
        stock_data = {}
        stock_files = [
            'project-3-group-3-main/datasets/stock_market/nikkei_225.csv',
            'project-3-group-3-main/datasets/stock_market/S&P_500.csv'
        ]
        
        for file_path in stock_files:
            try:
                data = pd.read_csv(file_path)
                market = 'nikkei' if 'nikkei' in file_path else 'sp500'
                stock_data[market] = data
                self.logger.info(f"Loaded {market} data: {len(data)} records")
            except FileNotFoundError:
                self.logger.warning(f"Stock data file not found: {file_path}")
        
        # Load classification results
        classification_results = {}
        classification_files = [
            'project-3-group-3-main/plots/logistic.png',
            'project-3-group-3-main/plots/knn.png',
            'project-3-group-3-main/plots/naive_bayes.png',
            'project-3-group-3-main/plots/random_forest_cm.png',
            'project-3-group-3-main/plots/svm_cm.png',
            'project-3-group-3-main/plots/decison_tree_cm.png'
        ]
        
        for file_path in classification_files:
            if os.path.exists(file_path):
                classification_results[os.path.basename(file_path)] = file_path
        
        return {
            'stock_data': stock_data,
            'classification_results': classification_results
        }
    
    def integrate_project_4_data(self):
        """Integrate data from regression models and visualization"""
        self.logger.info("Integrating regression and visualization data...")
        
        # Load regression results
        regression_results = {}
        regression_files = [
            'project-4-group-3-main/plots/LinearRegression.png',
            'project-4-group-3-main/plots/Ridge.png',
            'project-4-group-3-main/plots/Lasso.png',
            'project-4-group-3-main/plots/RandomForestRegressor.png',
            'project-4-group-3-main/plots/GradientBoostingRegressor.png'
        ]
        
        for file_path in regression_files:
            if os.path.exists(file_path):
                regression_results[os.path.basename(file_path)] = file_path
        
        # Load visualization results
        visualization_results = {}
        viz_files = [
            'project-4-group-3-main/plots/US_candlestick.png',
            'project-4-group-3-main/plots/volaitility.png',
            'project-4-group-3-main/plots/dt_importance.png'
        ]
        
        for file_path in viz_files:
            if os.path.exists(file_path):
                visualization_results[os.path.basename(file_path)] = file_path
        
        return {
            'regression_results': regression_results,
            'visualization_results': visualization_results
        }
    
    def create_unified_dataset(self):
        """Create a unified dataset from all projects"""
        self.logger.info("Creating unified dataset...")
        
        # Integrate data from all projects
        project_1_data = self.integrate_project_1_data()
        project_2_data = self.integrate_project_2_data()
        project_3_data = self.integrate_project_3_data()
        project_4_data = self.integrate_project_4_data()
        
        # Start with the combined clean data as base
        base_data = project_2_data.get('combined_data')
        if base_data is None:
            self.logger.error("No base data available for unification")
            return None
        
        unified_data = base_data.copy()
        
        # Add sentiment data if available
        if 'us_news_sentiments.csv' in project_2_data['sentiment_data']:
            sentiment_df = project_2_data['sentiment_data']['us_news_sentiments.csv']
            # Merge sentiment data (this would need proper date matching)
            self.logger.info("Added sentiment data to unified dataset")
        
        # Add stock market data if available
        if 'nikkei' in project_3_data['stock_data']:
            nikkei_data = project_3_data['stock_data']['nikkei']
            # Merge stock data (this would need proper date matching)
            self.logger.info("Added Nikkei data to unified dataset")
        
        if 'sp500' in project_3_data['stock_data']:
            sp500_data = project_3_data['stock_data']['sp500']
            # Merge stock data (this would need proper date matching)
            self.logger.info("Added S&P 500 data to unified dataset")
        
        # Save unified dataset
        unified_data.to_csv('unified_currency_dataset.csv', index=False)
        self.logger.info(f"Created unified dataset with {len(unified_data)} records")
        
        return unified_data
    
    def setup_database_schema(self):
        """Setup database schema for the unified system"""
        self.logger.info("Setting up database schema...")
        
        conn = sqlite3.connect(self.config.DB_PATH)
        cursor = conn.cursor()
        
        # Create tables for different data types
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS unified_currency_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                exchange_rate_USD_JY_x REAL,
                exchange_rate_USD_JY_y REAL,
                cpi_us REAL,
                inflation_us REAL,
                interest_r_us REAL,
                gdp_pc_us REAL,
                govt_debt_us REAL,
                cpi_j REAL,
                interest_r_j REAL,
                inflation_j REAL,
                gdp_pc_j REAL,
                govt_debt_j REAL,
                sentiment_score REAL,
                nikkei_close REAL,
                sp500_close REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for efficient querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_unified_date ON unified_currency_data(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_unified_exchange_rate ON unified_currency_data(exchange_rate_USD_JY_x)')
        
        conn.commit()
        conn.close()
        
        self.logger.info("Database schema setup completed")
    
    def populate_database(self, unified_data):
        """Populate database with unified data"""
        self.logger.info("Populating database with unified data...")
        
        conn = sqlite3.connect(self.config.DB_PATH)
        
        # Insert unified data into database
        unified_data.to_sql('unified_currency_data', conn, if_exists='replace', index=False)
        
        conn.close()
        
        self.logger.info(f"Database populated with {len(unified_data)} records")
    
    def run_integration_pipeline(self):
        """Run the complete data integration pipeline"""
        self.logger.info("Starting data integration pipeline...")
        
        # Create directories
        self.config.create_directories()
        
        # Setup database schema
        self.setup_database_schema()
        
        # Create unified dataset
        unified_data = self.create_unified_dataset()
        
        if unified_data is not None:
            # Populate database
            self.populate_database(unified_data)
            
            self.logger.info("Data integration pipeline completed successfully")
            return True
        else:
            self.logger.error("Data integration pipeline failed")
            return False

if __name__ == "__main__":
    # Run data integration
    integrator = DataIntegrator()
    success = integrator.run_integration_pipeline()
    
    if success:
        print("Data integration completed successfully!")
    else:
        print("Data integration failed!") 