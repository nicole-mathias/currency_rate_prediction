#!/bin/bash
# Deploy to Render - Free Platform
# No trial period, always free

echo "🚀 Deploying Currency Prediction System to Render"
echo "=================================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit for deployment"
fi

# Check if remote exists
if ! git remote | grep -q origin; then
    echo "Please add your GitHub repository as origin:"
    echo "git remote add origin https://github.com/yourusername/your-repo.git"
    echo "git push -u origin main"
    exit 1
fi

# Push to GitHub
echo "Pushing to GitHub..."
git add .
git commit -m "Deploy to Render - $(date)"
git push origin main

echo ""
echo "✅ Code pushed to GitHub!"
echo ""
echo "Next steps:"
echo "1. Go to https://render.com"
echo "2. Sign up with GitHub"
echo "3. Click 'New +' → 'Web Service'"
echo "4. Connect your repository"
echo "5. Use these settings:"
echo "   - Name: currency-prediction-system"
echo "   - Environment: Python 3"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: gunicorn app:app"
echo "6. Click 'Create Web Service'"
echo ""
echo "Your app will be available at: https://currency-prediction-system.onrender.com"
echo ""
echo "Note: Render is always free with no trial period!"
