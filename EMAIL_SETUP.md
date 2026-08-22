# Email Configuration Guide

DeepSentinel uses **SendGrid** for fraud alert emails. No passwords needed - just an API key.

**Security:** We never store user passwords. Only SendGrid API key required.

## Setup Steps

### Step 1: Create Free SendGrid Account

1. Go to https://sendgrid.com
2. Click **Sign Up for Free**
3. Complete the registration
4. Verify your email

### Step 2: Create API Key

1. Go to https://app.sendgrid.com/settings/api_keys
2. Click **Create API Key**
3. Name it: `DeepSentinel-Fraud-Alerts`
4. Copy the key (you'll only see it once!)

### Step 3: Configure .env

Create/edit `.env` in DeepSentinel root:

```ini
SENDGRID_API_KEY=SG.xxxxx_paste_your_key_here_xxxxx
SENDER_EMAIL=alerts@deepsentinel.io
SENDER_NAME=DeepSentinel Fraud Alerts
```

### Step 4: Restart Backend

```bash
cd C:\Projects\DeepSentinel
pip install -r requirements.txt  # First time only
python -m uvicorn backend.main:app --reload
```

✅ Backend running at http://localhost:8000

---

## Email Features

### Settings Page

http://localhost:3000/settings

- Add/remove risk managers
- Configure alert thresholds
- Send test emails
- Preview email templates

### Email Template Preview

View how fraud alerts look:

```
http://localhost:8000/email-template/preview?classification=HIGH
```

Classifications: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`

### API Endpoints

**Get Settings:**
```bash
curl http://localhost:8000/settings
```

**Add Risk Manager:**
```bash
curl -X POST http://localhost:8000/settings/risk-manager \
  -H "Content-Type: application/json" \
  -d '{"name":"Alice","email":"alice@example.com","role":"Analyst"}'
```

**Send Test Email:**
```bash
curl -X POST http://localhost:8000/email/send-test \
  -H "Content-Type: application/json" \
  -d '{"name":"Test Manager","email":"test@gmail.com"}'
```

**Remove Risk Manager:**
```bash
curl -X DELETE http://localhost:8000/settings/risk-manager/alice@example.com
```

---

## Email Template Structure

When a fraud transaction is detected, an email is automatically sent containing:

1. **Risk Badge** — CRITICAL/HIGH/MEDIUM/LOW with confidence score
2. **Transaction Details** — ID and timestamp
3. **AI Model Scores** — Graph/Behavioral/Temporal scores with forensic signals
4. **FATF Typology Match** — Retrieved fraud pattern classification
5. **LLM Forensic Analysis** — AI-generated forensic report
6. **Action Link** — Direct link to review in DeepSentinel dashboard

---

## Troubleshooting

**"Email send failed. Check Gmail app password or SendGrid API key"**
- Verify GMAIL_ADDRESS and GMAIL_APP_PASSWORD are correct
- Ensure app password has spaces removed
- Check firewall allows SMTP on port 465
- Restart backend after changing .env

**"No recipient emails configured"**
- Add at least one risk manager via Settings page or `/settings/risk-manager` endpoint

**Emails going to Spam**
- Gmail may temporarily filter new senders
- Add DeepSentinel to contacts to whitelist
- SendGrid has higher deliverability rates for production

**SMTP Connection Timeout**
- Check internet connection
- Verify firewall allows outbound SMTP (port 465)
- Try with SendGrid if Gmail times out
