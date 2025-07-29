#!/usr/bin/env python3
"""
Currency Rate Prediction System - Main Launcher
==============================================

This is the main entry point for the unified currency prediction system.
It orchestrates all components: data collection, processing, ML models, and API.
"""

import sys
import os
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from data_processing.data_integration import DataIntegrator
from api_dashboard.currency_prediction_system_simple import CurrencyPredictionSystem

def setup_logging():
    """Setup logging for the system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/system_launcher.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main launcher function"""
    logger = setup_logging()
    
    print("=" * 60)
    print("Currency Rate Prediction System")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()
    
    try:
        # Initialize configuration
        config = Config()
        config.create_directories()
        
        print("Created necessary directories")
        
        # Run data integration
        print("\nRunning data integration...")
        integrator = DataIntegrator()
        integration_success = integrator.run_integration_pipeline()
        
        if integration_success:
            print("Data integration completed successfully!")
        else:
            print("Data integration completed with warnings")
        
        # Initialize the main system
        print("\nInitializing currency prediction system...")
        system = CurrencyPredictionSystem(config)
        
        print("\nSystem components:")
        print("   • Data Collection: Ready")
        print("   • Feature Engineering: Ready")
        print("   • Ensemble Models: Ready")
        print("   • Sentiment Analysis: Ready")
        print("   • Database Management: Ready")
        print("   • Flask API: Ready")
        
        print("\nAvailable features:")
        print("   • Real-time currency predictions")
        print("   • Multi-source data collection")
        print("   • Advanced technical indicators")
        print("   • Sentiment analysis pipeline")
        print("   • Interactive dashboard")
        
        print("\nStarting API server...")
        print(f"   • API URL: http://localhost:{config.API_PORT}")
        print(f"   • Dashboard: http://localhost:{config.API_PORT}")
        print("   • API Endpoints:")
        print("     - GET /api/predict/<currency_pair>")
        print("     - GET /api/sentiment/<currency_pair>")
        print("     - GET /api/performance")
        
        print("\nSample API calls:")
        print(f"   • curl http://localhost:{config.API_PORT}/api/predict/USDJPY")
        print(f"   • curl http://localhost:{config.API_PORT}/api/sentiment/USDJPY")
        print(f"   • curl http://localhost:{config.API_PORT}/api/performance")
        
        print("\n" + "=" * 60)
        print("System is ready! Press Ctrl+C to stop.")
        print("=" * 60)
        
        # Start the API server
        print(f"\nAPI Server starting on port {config.API_PORT}")
        print(f"Dashboard available at: http://localhost:{config.API_PORT}")
        print(f"API endpoints:")
        print(f"   • Predictions: http://localhost:{config.API_PORT}/api/predict/USDJPY")
        print(f"   • Sentiment: http://localhost:{config.API_PORT}/api/sentiment/USDJPY")
        print(f"   • Performance: http://localhost:{config.API_PORT}/api/performance")
        
        # Use environment variable for port in deployment
        port = int(os.environ.get('PORT', config.API_PORT))
        system.start_api(port=port)
        
    except KeyboardInterrupt:
        print("\n\nSystem stopped by user")
        logger.info("System stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        logger.error(f"System error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 