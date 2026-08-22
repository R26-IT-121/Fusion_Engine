# DeepSentinel — Quick Start Guide

Get the fusion engine + web app running in minutes.

---

## Step 1: Create config.ini

```powershell
cd C:\Projects\DeepSentinel
Copy-Item config.example.ini config.ini
```

Open `config.ini` and fill in the `[secrets]` section:

```ini
[secrets]
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
jwt_secret_key = <paste generated key>

# https://aistudio.google.com/app/apikey
gemini_api_key = <your key>

# https://app.sendgrid.com/settings/api_keys  (optional until you test email)
sendgrid_api_key =
```

Everything else has a working default. `config.ini` is gitignored — it holds
real keys.

**Precedence:** environment variable > `config.ini` > built-in default. That is
how cloud deploys inject secrets without editing files.

---

## Step 2: Start Backend (Terminal 1)

```powershell
cd C:\Projects\DeepSentinel

# Install dependencies (first time only).
# Note: `pip` alone is not on PATH here — use `python -m pip`.
python -m pip install -r requirements.txt

# Or, preferred, from the lock file:
#   python -m poetry install

# Start API server
python -m uvicorn backend.main:app --reload --port 8000
```

On startup the resolved configuration is printed to the log with secrets
masked — use it to confirm which values are actually in effect.

Backend running on: http://localhost:8000

---

## Step 3: Start Web App (Terminal 2)

```powershell
cd C:\Projects\Deepsentinel-WEB

# Install dependencies (first time only)
npm install

# Start dev server
npm run dev
```

Web app running on: http://localhost:5173

---

## Step 4: Log In

1. Open http://localhost:5173/login
2. Username `admin`, password `admin123`
   (or whatever `[secrets] admin_bootstrap_password` is set to)

The admin account is created on first backend start, when `users.json` does
not yet exist. Change the password before deploying anywhere.

**Roles:**

| Role | Intended for | Access |
|------|--------------|--------|
| `admin` | DeepSentinel team | Everything, including configuration |
| `risk_manager` | Bank risk manager | Transactions, monitoring, alerts — not configuration |
| `analyst` | Bank assistant manager | Read-only: view transactions and reports |

---

## Step 5: Test Email

### 5a. Add a Risk Manager

1. Open http://localhost:5173/settings
2. Under **Add New Risk Manager**:
   - Name: `Test Manager`
   - Email: your address
   - Role: `Risk Manager`
3. Click **Add Manager**

### 5b. Send a Test Email

1. Find the risk manager in the list
2. Click **Test**
3. Check the inbox (and spam folder)

With no SendGrid key configured, the send is mocked and logged rather than
delivered — useful for checking the flow without an account.

### 5c. View the Email Template

Click **Preview Email Template**, or open:

```
http://localhost:8000/email-template/preview?classification=HIGH
```

Classifications: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`.

---

## Step 6: Test the Pipeline

1. Open http://localhost:5173/analyzer
2. Select a fraud scenario (e.g. "Mule Network")
3. Click **Run Pipeline**
4. Watch it move through: Submit → Score → Fuse → Retrieve → Report

**Expected output:**

- Fraud confidence score (0–100%)
- Per-modality scores (Graph / Behavioral / Temporal)
- FATF typology match
- LLM forensic report
- Ablation comparison (RAG vs no-RAG)

Upstream scores are simulated until M1 and M3 ship their APIs. The fusion
engine, RAG retrieval, and LLM report are live.

---

## Step 7: API Documentation

```
http://localhost:8000/docs
```

**Key endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `POST /auth/login` | Log in, returns a JWT |
| `GET /auth/me` | Current user |
| `POST /analyze` | Analyze a transaction or scenario |
| `GET /settings` | Read configuration |
| `POST /settings/risk-manager` | Add a risk manager |
| `DELETE /settings/risk-manager/{email}` | Remove a risk manager |
| `POST /email/send-test` | Send a test email |
| `GET /email-template/preview` | View the email template |

---

## Troubleshooting

### "pip is not recognized"

`pip` is not on PATH on this machine. Use `python -m pip` instead.

### numpy or scikit-learn fails to build

Older pinned versions have no Python 3.13 wheels, so pip falls back to
compiling from source and fails without a C compiler. `requirements.txt` now
uses ranges rather than hard pins, which resolves to versions that do ship
wheels. If you still hit it, `python -m poetry install` uses the lock file.

### Email send failed

- Check `[secrets] sendgrid_api_key` in `config.ini`
- `[email] sender_email` must be a **verified sender** in your SendGrid
  account, otherwise SendGrid rejects the send
- Restart the backend after editing `config.ini`

### A config change had no effect

An environment variable overrides `config.ini`. The startup log prints the
resolved configuration — it shows the value actually in use. If a stale value
persists, look for it in `.env` or your shell environment.

### Backend won't start

```powershell
Remove-Item -Path "C:\Projects\DeepSentinel\backend\__pycache__" -Recurse -Force
python -m pip install --upgrade -r requirements.txt
python -m uvicorn backend.main:app --reload
```

### Web app won't load

```powershell
Remove-Item -Path "C:\Projects\Deepsentinel-WEB\node_modules" -Recurse -Force
npm install
npm run dev
```

---

## Next Steps (Aug 24–28)

| Date | Task |
|------|------|
| **Aug 24** | M1 and M3 APIs arrive → update `[upstream]` in `config.ini` → test full pipeline |
| **Aug 25** | Enforce auth on API endpoints, transaction input form, tests |
| **Aug 26** | PostgreSQL migration, cloud deployment |
| **Aug 27** | Mobile app, email automation, UI polish |
| **Aug 28** | Final demo |

---

## Files Reference

| Path | Purpose |
|------|---------|
| `config.ini` | All configuration (gitignored) |
| `config.example.ini` | Documented template |
| `backend/config.py` | Config loader: env > config.ini > default |
| `backend/main.py` | FastAPI server and endpoints |
| `backend/auth.py` | Authentication and role-based access control |
| `backend/adapters/upstream.py` | M1/M2/M3 API integration |
| `backend/email_service.py` | Fraud alert email via SendGrid |
| `backend/settings.py` | Risk manager and alert configuration |
| `backend/rag/` | RAG pipeline and prompt construction |
| `pyproject.toml` / `poetry.lock` | Dependencies, reproducible builds |

---

## Getting Help

- **API errors** — check http://localhost:8000/docs
- **UI issues** — check the browser console (F12)
- **Email problems** — see EMAIL_SETUP.md
- **Config questions** — see the comments in config.example.ini
