# 🚀 FREE DEPLOYMENT INSTRUCTIONS

## **Option 1: RENDER (Recommended - Always Free)**

### **Step 1: Prepare Your Repository**
1. Push your code to GitHub (if not already done):
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

### **Step 2: Deploy to Render**
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name**: `currency-prediction-system`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Port**: `10000` (Render will set this automatically)

### **Step 3: Environment Variables**
Add these in Render dashboard:
- `PORT`: `10000`
- `PYTHON_VERSION`: `3.11.7`

### **Step 4: Deploy**
Click "Create Web Service" and wait for deployment (5-10 minutes)

---

## **Option 2: PYTHONANYWHERE (Always Free)**

### **Step 1: Upload Files**
1. Go to [pythonanywhere.com](https://pythonanywhere.com)
2. Sign up for free account
3. Go to "Files" tab
4. Upload your project files

### **Step 2: Create Web App**
1. Go to "Web" tab
2. Click "Add a new web app"
3. Choose "Flask"
4. Select Python 3.10
5. Set path to your `app.py`

### **Step 3: Configure**
1. Set source code path to your project folder
2. Set working directory to your project folder
3. Set WSGI file to `app.py`
4. Reload web app

---

## **Option 3: FLY.IO (Always Free)**

### **Step 1: Install Fly CLI**
```bash
# macOS
brew install flyctl

# Or download from https://fly.io/docs/hands-on/install-flyctl/
```

### **Step 2: Create Fly App**
```bash
flyctl auth login
flyctl launch
```

### **Step 3: Deploy**
```bash
flyctl deploy
```

---

## **Option 4: HEROKU (Low Cost - $5/month)**

### **Step 1: Install Heroku CLI**
```bash
# macOS
brew install heroku/brew/heroku
```

### **Step 2: Create Heroku App**
```bash
heroku login
heroku create your-currency-app-name
```

### **Step 3: Deploy**
```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

---

## **RECOMMENDED: RENDER**

**Why Render?**
- ✅ Always free (no trial period)
- ✅ Automatic deployments from GitHub
- ✅ Custom domains
- ✅ SSL certificates
- ✅ Easy to use
- ✅ Reliable uptime

**Limitations:**
- Sleeps after 15 minutes of inactivity
- 750 hours/month free
- Wakes up in ~30 seconds when accessed

---

## **TROUBLESHOOTING**

### **Common Issues:**
1. **Port Issues**: Make sure your app uses `os.environ.get('PORT', 8080)`
2. **Dependencies**: Ensure all packages are in `requirements.txt`
3. **File Paths**: Use relative paths, not absolute paths
4. **Data Files**: Large data files might need to be hosted separately

### **Your App is Ready:**
- ✅ `app.py` - Flask application
- ✅ `requirements.txt` - Dependencies
- ✅ `Procfile` - Process file
- ✅ `runtime.txt` - Python version
- ✅ `render.yaml` - Render configuration

**Next Steps:**
1. Choose Render (recommended)
2. Follow the Render deployment steps
3. Your app will be live at `https://your-app-name.onrender.com`
