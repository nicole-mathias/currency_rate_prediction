#!/usr/bin/env python3
"""
PythonAnywhere Deployment Helper
===============================

This script helps set up your Flask app on PythonAnywhere.
Run this locally to prepare your files for PythonAnywhere.
"""

import os
import shutil

def prepare_for_pythonanywhere():
    """Prepare files for PythonAnywhere deployment"""
    
    print("🐍 Preparing for PythonAnywhere Deployment")
    print("=" * 50)
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ app.py not found!")
        return False
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found!")
        return False
    
    # Create a simple WSGI file for PythonAnywhere
    wsgi_content = '''import sys
import os

# Add your project directory to the Python path
path = '/home/yourusername/currency_rate_prediction'
if path not in sys.path:
    sys.path.append(path)

from app import app as application

if __name__ == "__main__":
    application.run()
'''
    
    with open('wsgi.py', 'w') as f:
        f.write(wsgi_content)
    
    print("✅ Created wsgi.py for PythonAnywhere")
    
    # Create a simple startup script
    startup_script = '''#!/bin/bash
# PythonAnywhere startup script

echo "Starting Currency Prediction System..."

# Install dependencies
pip3.10 install --user -r requirements.txt

# Run the app
python3.10 app.py
'''
    
    with open('start.sh', 'w') as f:
        f.write(startup_script)
    
    os.chmod('start.sh', 0o755)
    print("✅ Created start.sh startup script")
    
    # Create a simple requirements file for PythonAnywhere
    pa_requirements = '''Flask==3.0.0
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
xgboost==2.0.3
plotly==5.17.0
yfinance==0.2.28
textblob==0.17.1
gunicorn==21.2.0
'''
    
    with open('requirements_pa.txt', 'w') as f:
        f.write(pa_requirements)
    
    print("✅ Created requirements_pa.txt for PythonAnywhere")
    
    print("\n📋 PythonAnywhere Deployment Steps:")
    print("1. Go to https://pythonanywhere.com")
    print("2. Sign up for free account")
    print("3. Go to 'Files' tab")
    print("4. Upload all your project files")
    print("5. Go to 'Web' tab")
    print("6. Click 'Add a new web app'")
    print("7. Choose 'Flask' and Python 3.10")
    print("8. Set source code path to your project folder")
    print("9. Set working directory to your project folder")
    print("10. Set WSGI file to 'wsgi.py'")
    print("11. Reload web app")
    print("\nYour app will be available at: https://yourusername.pythonanywhere.com")
    
    return True

if __name__ == "__main__":
    prepare_for_pythonanywhere()
