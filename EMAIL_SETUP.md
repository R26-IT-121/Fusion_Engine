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

### Step 3: Configure config.ini

Edit `config.ini` in the DeepSentinel root:

```ini
[email]
sender_email = alerts@deepsentinel.io
sender_name = DeepSentinel Fraud Alerts

[secrets]
sendgrid_api_key = SG.xxxxx_paste_your_key_here_xxxxx
```

`sender_email` must be a **verified sender** in your SendGrid account, or
SendGrid will reject the send.

In production, inject `SENDGRID_API_KEY` as an environment variable through
your platform's secret manager instead — it overrides `config.ini`.

### Step 4: Restart Backend

```bash
cd C:\Projects\DeepSentinel
python -m pip install -r requirements.txt   # First time only
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

**"Email send failed"**
- Verify `[secrets] sendgrid_api_key` in `config.ini` is set
- Confirm `[email] sender_email` is a verified sender in SendGrid —
  an unverified sender is the most common cause of a 403
- Restart the backend after editing `config.ini`
- Check the backend log: the SendGrid status code and response body are logged

**Nothing sent, but no error either**
- With no API key configured, sends are mocked and logged rather than
  delivered. Look for `Mock email send (SendGrid not configured)` in the log.

**"No recipient emails configured"**
- Add at least one risk manager via the Settings page or the
  `/settings/risk-manager` endpoint

**Emails going to Spam**
- Expected for a new sending domain
- Set up domain authentication (SPF/DKIM) in SendGrid for production
  deliverability
- Try with SendGrid if Gmail times out
