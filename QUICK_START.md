# ⚡ QUICK START - Everything You Need

## ✅ What Just Happened

1. **✅ Pushed to GitHub:** https://github.com/nargesghv/Helpdesk-Agent
2. **✅ Ready for Netlify:** Can deploy frontend now
3. **✅ Backend scripts ready:** Easy local setup

---

## 🚀 Run Backend Locally (3 Steps)

### Step 1: Activate Virtual Environment
```powershell
cd "C:\Users\nariv\Desktop\Git\helpdesk agent"
.\agent\Scripts\Activate.ps1
```
*You should see `(agent)` at the start of your prompt*

### Step 2: Set Groq API Key
```powershell
$env:GROQ_API_KEY = "your-groq-api-key-here"
```
*Get free key: https://console.groq.com/keys*

### Step 3: Run Server
```powershell
python -m uvicorn backend:app --host 0.0.0.0 --port 8004 --reload
```

**OR use the setup script:**
```powershell
.\setup_backend.ps1
```

### Expected Output
```
✅ Using Groq model: llama-3.1-8b-instant
✅ Graph initialized
INFO: Uvicorn running on http://0.0.0.0:8004
INFO: Application startup complete.
```

**Then open:** http://localhost:8004

---

## 🌐 Deploy Frontend to Netlify

### Option 1: Via Netlify Dashboard (EASIEST)

1. Go to https://app.netlify.com/teams/nargesghv/projects
2. Click "Add new site" → "Import an existing project"
3. Choose "Deploy with GitHub"
4. Select: `nargesghv/Helpdesk-Agent`
5. Settings:
   - **Build command:** (leave empty)
   - **Publish directory:** `.`
6. Click "Deploy site"

**Done!** Your frontend will be at: `https://your-site-name.netlify.app`

### Option 2: Via Netlify CLI

```powershell
# Install Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Deploy
netlify deploy --prod
```

---

## 🎯 Complete Workflow

```
┌─────────────────────────────────────┐
│  1. Activate Virtual Environment    │
│     .\agent\Scripts\Activate.ps1    │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  2. Set Groq API Key                │
│     $env:GROQ_API_KEY = "key"       │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  3. Run Backend                     │
│     python -m uvicorn backend:app   │
│     --host 0.0.0.0 --port 8004      │
│     --reload                        │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  4. Open Browser                    │
│     http://localhost:8004           │
└─────────────────────────────────────┘
```

---

## 📝 Daily Use

### Start Working
```powershell
cd "C:\Users\nariv\Desktop\Git\helpdesk agent"
.\agent\Scripts\Activate.ps1
python -m uvicorn backend:app --host 0.0.0.0 --port 8004 --reload
```

### Push Updates
```powershell
git add .
git commit -m "Your update message"
git push
```
*Netlify auto-deploys if connected to GitHub*

---

## 🐛 Troubleshooting

### "No module named uvicorn"
**Fix:** Activate virtual environment first
```powershell
.\agent\Scripts\Activate.ps1
```

### "Cannot find file specified" (pip error)
**Fix:** Use python -m pip
```powershell
.\agent\Scripts\Activate.ps1
python -m pip install fastapi uvicorn[standard]
```

### CORS Errors (Frontend can't reach backend)
**Fix:** Make sure backend is running and accessible
```powershell
# Check backend is running
curl http://localhost:8004/health

# If not, start it
.\agent\Scripts\Activate.ps1
python -m uvicorn backend:app --host 0.0.0.0 --port 8004 --reload
```

---

## 📚 Useful Links

- **GitHub Repo:** https://github.com/nargesghv/Helpdesk-Agent
- **Netlify Dashboard:** https://app.netlify.com/teams/nargesghv/projects
- **Get Groq API Key:** https://console.groq.com/keys
- **Full Documentation:** See `DEPLOY_GUIDE.md`

---

## 🎓 Architecture

```
Frontend (Netlify)              Backend (Your PC)
     │                                │
     │  HTTP Requests                 │
     ├───────────────────────────────>│
     │                                │
     │  • Submit tickets              │ • LangGraph Pipeline
     │  • Upload files                │ • FAISS + BM25 Search
     │  • View results                │ • Groq LLM Generation
     │                                │ • Process tickets
     │<───────────────────────────────┤
     │  JSON Responses                │
```

---

## ✅ Checklist

Before deploying:
- [x] Code pushed to GitHub ✅
- [ ] Groq API key obtained
- [ ] Virtual environment activated
- [ ] Backend running locally
- [ ] Frontend deployed to Netlify

---

## 🚀 You're Ready!

**Everything is set up!** Just follow the steps above to:
1. Run backend locally
2. Deploy frontend to Netlify
3. Start processing tickets!

---

**Need help?** Check `DEPLOY_GUIDE.md` for detailed instructions.

**Quick command to start:**
```powershell
.\setup_backend.ps1
```

🎉 **Happy coding!**

