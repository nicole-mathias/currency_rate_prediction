#!/usr/bin/env python3
"""
Run Tests in Virtual Environment
===============================

This script activates the virtual environment and runs the system tests
to get real performance metrics and improvement numbers.
"""

import os
import sys
import subprocess
import platform

def activate_venv_and_run_tests():
    """Activate virtual environment and run tests"""
    print("🚀 Running Enhanced Currency Prediction System Tests")
    print("=" * 60)
    
    # Determine the activation script path
    if platform.system() == "Windows":
        activate_script = "venv\\Scripts\\activate"
        python_path = "venv\\Scripts\\python"
    else:
        activate_script = "venv/bin/activate"
        python_path = "venv/bin/python"
    
    # Check if virtual environment exists
    if not os.path.exists("venv"):
        print("❌ Virtual environment not found!")
        print("Please run: python -m venv venv")
        return 1
    
    try:
        print("📦 Installing requirements...")
        
        # Install requirements
        if platform.system() == "Windows":
            subprocess.run([python_path, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        else:
            subprocess.run([python_path, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        
        print("✅ Requirements installed!")
        
        print("\n🧪 Running system tests...")
        
        # Run the test script
        result = subprocess.run([python_path, "test_system.py"], check=True)
        
        if result.returncode == 0:
            print("\n✅ All tests completed successfully!")
            print("\n📊 Check the following files for results:")
            print("  • performance_report.json - Detailed performance metrics")
            print("  • test_results.log - Test execution logs")
            
            # Display summary if performance report exists
            if os.path.exists("performance_report.json"):
                import json
                with open("performance_report.json", "r") as f:
                    report = json.load(f)
                
                print("\n📈 PERFORMANCE SUMMARY:")
                print("-" * 40)
                
                if "improvements" in report:
                    for key, value in report["improvements"].items():
                        if "improvement" in key:
                            print(f"  {key}: {value:.2f}%")
                
                if "ensemble_models" in report:
                    print("\n🤖 Ensemble Models Performance:")
                    for model, metrics in report["ensemble_models"].items():
                        print(f"  {model}:")
                        print(f"    MAE: {metrics.get('mae', 0):.4f}")
                        print(f"    R²: {metrics.get('r2', 0):.4f}")
                        if 'improvement_over_baseline' in metrics:
                            print(f"    Improvement: {metrics['improvement_over_baseline']:.2f}%")
            
            return 0
        else:
            print(f"❌ Tests failed with exit code: {result.returncode}")
            return result.returncode
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running tests: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

def main():
    """Main function"""
    return activate_venv_and_run_tests()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 