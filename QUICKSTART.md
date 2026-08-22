# DeepSentinel — Quick Start Guide

Get the fusion engine + web app running in minutes.

---

## Step 1: Setup Gmail for Email Testing

### Generate App Password (required once)

1. Go to https://myaccount.google.com/apppasswords
2. Select **Mail** and **Windows Computer**
3. Click **Generate** and copy the 16-character password

### Create .env File

Create `C:\Projects\DeepSentinel\.env`:

```ini
# LLM (keep as is or use your own Gemini key)
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSyD...your_key_here

# Email (use for testing)
GMAIL_ADDRESS=thiyaana.vidanaarachchi@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

# Upstream APIs (will add when M1/M3 arrive)
BEHAVIORAL_API_BASE=http://localhost:8001
GRAPH_API_BASE=http://localhost:8002
TEMPORAL_API_BASE=http://localhost:8003
```

---

## Step 2: Start Backend (Terminal 1)

```powershell
cd C:\Projects\DeepSentinel

# Install dependencies (first time only)
pip install -r requirements.txt

# Start API server
python -m uvicorn backend.main:app --reload --port 8000
```

✅ Backend running on: http://localhost:8000

---

## Step 3: Start Web App (Terminal 2)

```powershell
cd C:\Projects\Deepsentinel-WEB

# Install dependencies (first time only)
npm install

# Start dev server
npm run dev
```

✅ Web app running on: http://localhost:5173

---

## Step 4: Test Email (Web App)

### 4a. Add Risk Manager

1. Open http://localhost:5173/settings
2. **Add New Risk Manager:**
   - Name: `Test Manager`
   - Email: `thiyaana.vidanaarachchi@gmail.com`
   - Role: `Risk Manager`
3. Click **Add Manager**

### 4b. Send Test Email

1. Find the risk manager in the list
2. Click **📧 Test** button
3. Check your email inbox (or spam folder)

**Expected:** Beautiful HTML fraud alert email

### 4c. View Email Template

1. Click "Preview Email Template" button
2. Opens email preview in browser

---

## Step 5: Test Pipeline (Analyzer)

### 5a. Run Mock Scenario

1. Open http://localhost:5173/analyzer
2. Select a fraud scenario (e.g., "Mule Network")
3. Click **▶ Run Pipeline**
4. Watch it process through all 5 steps:
   - Submit → Score → Fuse → Retrieve → Report

### 5b. Expected Output

- **Fraud Confidence Score** (0-100%)
- **AI Model Scores** (Graph/Behavioral/Temporal)
- **FATF Typology Match** (fraud pattern classification)
- **LLM Forensic Report** (AI-generated analysis)
- **Ablation Comparison** (RAG vs no-RAG)

---

## Step 6: API Documentation

### Browse Interactive Docs

```
http://localhost:8000/docs
```

**Key Endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `POST /analyze` | Analyze transaction or scenario |
| `GET /settings` | Get all settings |
| `POST /settings/risk-manager` | Add risk manager |
| `DELETE /settings/risk-manager/{email}` | Remove risk manager |
| `POST /email/send-test` | Send test email |
| `GET /email-template/preview` | View email template |

---

## Troubleshooting

### "Gmail send failed"

**Solution:**
- Verify `GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD` in `.env`
- App password should have spaces (they're automatic)
- Restart backend after changing `.env`
- Check your Gmail security settings

### Backend won't start

**Solution:**
```powershell
# Clear Python cache
Remove-Item -Path "C:\Projects\DeepSentinel\backend\__pycache__" -Recurse -Force

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Try again
python -m uvicorn backend.main:app --reload
```

### Web app won't load

**Solution:**
```powershell
# Clear node cache
Remove-Item -Path "C:\Projects\Deepsentinel-WEB\node_modules" -Recurse -Force

# Reinstall
npm install

# Try again
npm run dev
```

### Email goes to Spam

**Solution:**
- Add `alerts@deepsentinel.io` or your Gmail to contacts
- Check spam folder during testing
- Move to inbox to whitelist sender
- Use SendGrid for production (better deliverability)

---

## Next Steps (Aug 24-28)

| Date | Task |
|------|------|
| **Aug 24** | M1 & M3 APIs arrive → update `.env` → test full pipeline |
| **Aug 25** | Add real transaction input form → enhanced visualization |
| **Aug 26** | Cloud deployment (AWS/GCP) → email notifications live |
| **Aug 27** | Mobile app (React Native) → polish UI |
| **Aug 28** | Final demo ready |

---

## Files Reference

| Path | Purpose |
|------|---------|
| `backend/main.py` | FastAPI server (endpoints) |
| `backend/adapters/upstream.py` | M1/M2/M3 API integration |
| `backend/email_service.py` | Email sending (Gmail/SendGrid) |
| `backend/settings.py` | Configuration management |
| `backend/rag/` | RAG pipeline + LLM |
| `src/pages/Analyzer.jsx` | Pipeline visualization |
| `src/pages/Settings.jsx` | Risk manager configuration |
| `src/services/api.js` | Backend API client |

---

## Asking for Help

- **API errors?** Check `http://localhost:8000/docs`
- **UI issues?** Check browser console (F12)
- **Email problems?** See EMAIL_SETUP.md
- **Code questions?** Check code comments in backend files

---

**You're all set! Let's build the best fraud detection system. 🚀**
