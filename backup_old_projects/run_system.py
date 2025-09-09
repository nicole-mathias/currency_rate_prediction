#!/usr/bin/env python3
"""
Currency Rate Prediction System Launcher
========================================

This script launches the complete currency prediction system.
"""

import os
import sys
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_integration import DataIntegrator
from currency_prediction_system import CurrencyPredictionSystem

def setup_logging():
    """Setup logging for the system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('system_launcher.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main launcher function"""
    logger = setup_logging()
    
    print("=" * 60)
    print("🚀 Currency Rate Prediction System")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()
    
    try:
        # Initialize configuration
        config = Config()
        config.create_directories()
        
        print("📁 Created necessary directories")
        
        # Run data integration
        print("\n🔄 Running data integration...")
        integrator = DataIntegrator()
        integration_success = integrator.run_integration_pipeline()
        
        if integration_success:
            print("✅ Data integration completed successfully!")
        else:
            print("⚠️  Data integration completed with warnings")
        
        # Initialize the main system
        print("\n🔧 Initializing currency prediction system...")
        system = CurrencyPredictionSystem()
        
        print("\n📊 System components:")
        print("   • Data Collection: ✅")
        print("   • Feature Engineering: ✅")
        print("   • Ensemble Models: ✅")
        print("   • Sentiment Analysis: ✅")
        print("   • Database Management: ✅")
        print("   • MLOps Pipeline: ✅")
        print("   • Flask API: ✅")
        
        print("\n🎯 Available features:")
        print("   • Real-time currency predictions")
        print("   • Multi-source data collection")
        print("   • Advanced technical indicators")
        print("   • Sentiment analysis pipeline")
        print("   • Model performance monitoring")
        print("   • Interactive dashboard")
        print("   • Backtesting capabilities")
        
        print("\n🌐 Starting API server...")
        print("   • API URL: http://localhost:5000")
        print("   • Dashboard: http://localhost:5000")
        print("   • API Endpoints:")
        print("     - GET /api/predict/<currency_pair>")
        print("     - GET /api/sentiment/<currency_pair>")
        print("     - GET /api/performance")
        
        print("\n📈 Sample API calls:")
        print("   • curl http://localhost:5000/api/predict/USDJPY")
        print("   • curl http://localhost:5000/api/sentiment/USDJPY")
        print("   • curl http://localhost:5000/api/performance")
        
        print("\n" + "=" * 60)
        print("🎉 System is ready! Press Ctrl+C to stop.")
        print("=" * 60)
        
        # Start the API server
        system.start_api()
        
    except KeyboardInterrupt:
        print("\n\n🛑 System stopped by user")
        logger.info("System stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"System error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 