#!/usr/bin/env python3
"""
Test script to verify the new project structure works correctly
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported correctly"""
    print("Testing imports...")
    
    try:
        from config.config import Config
        print("Config imported successfully")
        
        from data_processing.data_integration import DataIntegrator
        print("DataIntegrator imported successfully")
        
        from api_dashboard.currency_prediction_system_simple import CurrencyPredictionSystem
        print("CurrencyPredictionSystem imported successfully")
        
        return True
    except Exception as e:
        print(f"Import error: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from config.config import Config
        config = Config()
        
        # Test key attributes
        assert hasattr(config, 'API_PORT')
        assert hasattr(config, 'DB_PATH')
        assert hasattr(config, 'CURRENCY_PAIRS')
        
        print(f"Config loaded successfully")
        print(f"   • API Port: {config.API_PORT}")
        print(f"   • DB Path: {config.DB_PATH}")
        print(f"   • Currency Pairs: {len(config.CURRENCY_PAIRS)}")
        
        return True
    except Exception as e:
        print(f"Config error: {e}")
        return False

def test_data_integration():
    """Test data integration setup"""
    print("\nTesting data integration...")
    
    try:
        from data_processing.data_integration import DataIntegrator
        integrator = DataIntegrator()
        
        print("DataIntegrator initialized successfully")
        return True
    except Exception as e:
        print(f"Data integration error: {e}")
        return False

def test_system_initialization():
    """Test system initialization"""
    print("\nTesting system initialization...")
    
    try:
        from config.config import Config
        from api_dashboard.currency_prediction_system_simple import CurrencyPredictionSystem
        
        config = Config()
        system = CurrencyPredictionSystem(config)
        
        print("System initialized successfully")
        return True
    except Exception as e:
        print(f"System initialization error: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("Testing Currency Prediction System Structure")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_data_integration,
        test_system_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Structure is working correctly.")
        return 0
    else:
        print("Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 