#!/usr/bin/env python3
"""
Simple launcher for the Currency Prediction System
"""

import sys
import os
import subprocess

# Run the main system
if __name__ == "__main__":
    try:
        # Get the path to the main.py file
        main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'currency_prediction_system', 'main.py')
        
        if not os.path.exists(main_path):
            print(f"❌ Error: main.py not found at {main_path}")
            sys.exit(1)
        
        # Run the main.py script
        result = subprocess.run([sys.executable, main_path], cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'currency_prediction_system'))
        sys.exit(result.returncode)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you're running this from the correct directory")
        sys.exit(1) 