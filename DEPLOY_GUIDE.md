# 🚀 Deployment Guide

Complete guide to deploy your Helpdesk Agent to GitHub, Netlify, and run backend locally.

---

## 📦 Step 1: Push to GitHub

### Initialize Git and Push

```powershell
# Initialize git repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Helpdesk Agent with Frontend + Backend"

# Add your GitHub repository
git remote add origin https://github.com/nargesghv/Helpdesk-Agent.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## 🌐 Step 2: Deploy Frontend to Netlify

### Option A: Deploy via Netlify UI (EASIEST)

1. **Go to** https://app.netlify.com/teams/nargesghv/projects

2. **Click** "Add new site" → "Import an existing project"

3. **Connect to GitHub**
   - Authorize Netlify to access your GitHub
   - Select repository: `nargesghv/Helpdesk-Agent`

4. **Configure build settings:**
   - **Build command:** (leave empty)
   - **Publish directory:** `.` (current directory)
   - Click "Deploy site"

5. **Your frontend is live!**
   - URL: `https://your-site-name.netlify.app`

### Option B: Deploy via Netlify CLI

```powershell
# Install Netlify CLI
npm install -g netlify-cli

# Login to Netlify
netlify login

# Deploy
cd "C:\Users\nariv\Desktop\Git\helpdesk agent"
netlify deploy --prod
```

---

## 💻 Step 3: Run Backend Locally

### Fix: Use Virtual Environment

Your issue is that you're using system Python instead of the virtual environment where packages are installed.

### Activate Virtual Environment

```powershell
# Navigate to project
cd "C:\Users\nariv\Desktop\Git\helpdesk agent"

# Activate virtual environment
.\agent\Scripts\Activate.ps1

# You should see (agent) at the start of your prompt
```

### Install Missing Packages (if needed)

```powershell
# Make sure you're in virtual environment (you should see (agent) prompt)
pip install fastapi uvicorn[standard] python-multipart
```

### Run Backend

```powershell
# Option 1: Using Python module
python -m uvicorn backend:app --host 0.0.0.0 --port 8004 --reload

# Option 2: Using run script
.\run.ps1

# Option 3: Direct Python
python backend.py
```

### Expected Output

```
Starting Helpdesk Agent on http://localhost:8004
INFO:     Uvicorn running on http://0.0.0.0:8004
✅ Using Groq model: llama-3.1-8b-instant
✅ Graph initialized
INFO:     Application startup complete.
```

---

## 🔧 Step 4: Connect Frontend to Backend

### For Local Testing

Frontend is configured to auto-detect:
- When accessed via `localhost` → connects to `http://localhost:8004`
- When deployed on Netlify → connects to `http://localhost:8004` (you need to run backend locally)

### For Production Backend (Optional)

If you want to deploy backend to a cloud service:

1. **Edit `frontend.html` line 379:**
   ```javascript
   const API_URL = window.location.hostname === 'localhost' 
       ? 'http://localhost:8004' 
       : 'https://your-backend-url.com';  // ← Change this
   ```

2. **Commit and push:**
   ```powershell
   git add frontend.html
   git commit -m "Update backend URL"
   git push
   ```

3. **Netlify will auto-deploy** (if connected to GitHub)

---

## 🎯 Complete Workflow

### Development (Local)

1. **Start backend:**
   ```powershell
   .\agent\Scripts\Activate.ps1
   python -m uvicorn backend:app --host 0.0.0.0 --port 8004 --reload
   ```

2. **Open frontend:**
   - Local: http://localhost:8004
   - Or deployed: https://your-site.netlify.app

### Production

1. **Frontend:** Deployed on Netlify (auto-updates from GitHub)
2. **Backend:** Running locally on your machine
3. **Access:** Open your Netlify URL, backend processes requests locally

---

## ⚙️ Configuration

### Set Groq API Key

```powershell
# Set for current session
$env:GROQ_API_KEY = "your-groq-api-key-here"

# Set permanently (Windows)
[System.Environment]::SetEnvironmentVariable('GROQ_API_KEY', 'your-key', 'User')
```

Get free key: https://console.groq.com/keys

---

## 🐛 Troubleshooting

### "No module named uvicorn"

**Problem:** Not using virtual environment

**Solution:**
```powershell
.\agent\Scripts\Activate.ps1  # You should see (agent) in prompt
python -m uvicorn backend:app --host 0.0.0.0 --port 8004 --reload
```

### "Fatal error in launcher: Unable to create process"

**Problem:** Old/broken pip installation

**Solution:** Use Python module directly
```powershell
.\agent\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### "UnicodeDecodeError"

**Already fixed!** The `backend.py` now uses UTF-8 encoding.

### CORS Errors on Netlify

**Problem:** Browser blocking requests from Netlify to localhost

**Solution:** Either:
1. Run frontend locally: `http://localhost:8004`
2. Or deploy backend to a cloud service

---

## 📁 Project Structure

```
Helpdesk-Agent/
├── frontend.html          # Web UI (deployed to Netlify)
├── index.html            # Netlify entry point
├── backend.py            # API server (run locally)
├── ticket_graph.py       # Core logic
├── requirements.txt      # Python dependencies
├── netlify.toml          # Netlify config
├── .gitignore           # Git ignore rules
├── Data/                # Historical tickets
├── input/               # New tickets (local only)
└── output/              # Results (local only)
```

---

## 🎓 Summary

✅ **Frontend:** Deploys to Netlify from GitHub  
✅ **Backend:** Runs on your local machine  
✅ **Git:** Code stored on GitHub  
✅ **Data:** Stays local (Data/, input/, output/)  

---

## 📖 Quick Commands Reference

```powershell
# Push to GitHub
git add .
git commit -m "Your message"
git push

# Run backend locally
.\agent\Scripts\Activate.ps1
python -m uvicorn backend:app --host 0.0.0.0 --port 8004 --reload

# Deploy to Netlify (auto if connected to GitHub)
# Or manually: netlify deploy --prod
```

---

**Your deployment is ready! Follow the steps above.** 🎉

For more details, check:
- GitHub: https://github.com/nargesghv/Helpdesk-Agent
- Netlify: https://app.netlify.com/teams/nargesghv/projects

