# Email Configuration Guide

DeepSentinel supports Gmail (for testing) and SendGrid (for production). Follow these steps to enable email fraud alerts.

## Option 1: Gmail SMTP (Testing)

### Step 1: Generate Gmail App Password

1. Go to https://myaccount.google.com/apppasswords
2. Select **Mail** and **Windows Computer** (or your device)
3. Click **Generate**
4. Copy the 16-character app password

### Step 2: Configure .env

Create a `.env` file in the DeepSentinel root directory:

```ini
GMAIL_ADDRESS=thiyaana.vidanaarachchi@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
```

### Step 3: Restart Backend

```bash
cd C:\Projects\DeepSentinel
python -m uvicorn backend.main:app --reload
```

### Step 4: Test Email

Go to http://localhost:8000/docs and try the `/email/send-test` endpoint:

```json
{
  "name": "Risk Manager",
  "email": "thiyaana.vidanaarachchi@gmail.com",
  "role": "Risk Manager"
}
```

**Note:** Emails may arrive in Spam folder initially.

---

## Option 2: SendGrid (Production)

### Step 1: Create SendGrid Account

1. Sign up at https://sendgrid.com
2. Create an API key at https://app.sendgrid.com/settings/api_keys
3. Copy the key

### Step 2: Configure .env

```ini
SENDGRID_API_KEY=SG.xxxxxxxxxxxx
SENDER_EMAIL=alerts@deepsentinel.io
SENDER_NAME=DeepSentinel
```

### Step 3: Restart Backend

```bash
python -m uvicorn backend.main:app --reload
```

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
