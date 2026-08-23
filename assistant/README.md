# Operator assistant (Professional package)

A tool-using agent with live access to the platform. Ask it about a transaction,
an account, or the state of the system and it calls the real services, then
answers from what came back.

Not to be confused with `chatbot/`, which is public and answers questions *about*
the research from documentation. This one *operates* the platform, so it is
gated, audited, and shows its evidence.

| | `chatbot/` | `assistant/` |
|---|---|---|
| Audience | site visitors | licensed operators |
| Knowledge | project documentation | live models, history, docs |
| Access | open | admin-enabled, entitled roles |
| Answers | cited passages | tool results + reasoning |

## Tools

| Tool | Does | Costly |
|---|---|---|
| `get_model_scores` | Scores a transaction through all three detectors | yes |
| `get_fraud_ring` | Pattern, sink account, convergence, top edges from GraphSAGE | yes |
| `search_analysis_history` | Past analyses by account, classification or score | no |
| `get_system_status` | Which upstream models are reachable | no |
| `search_documentation` | Methodology lookup (shares the chatbot index) | no |

"Costly" tools spend upstream model calls and can be switched off independently
of the assistant itself.

## How it reasons

Neither LLM backend exposes native function calling, so tool use runs over a
small JSON protocol:

```
{"tool": "get_fraud_ring", "arguments": {...}}   → executed, result fed back
{"answer": "..."}                                 → done
```

Gemini and Ollama are therefore interchangeable. Each run is bounded by
`max_steps`, and every step is returned in the response so the UI can show what
was actually run — an analyst acting on a fraud verdict needs the evidence, not
just a conclusion.

**Without a language model** the assistant still works: it keyword-routes to a
tool and returns the raw result. Less useful, but it keeps the feature
demonstrable offline and makes the tool layer testable on its own.

## Entitlement

Two independent checks, both in `entitlement.py`:

1. **Master switch** — `enabled`, off by default. An admin can disable the
   feature deployment-wide without a redeploy.
2. **Seat type** — `allowed_roles`. Defaults to admin and risk manager;
   analysts (base package) are excluded until an admin opts them in.

State persists in the runtime settings JSON the app already uses, so there is no
database migration and an operator can read it as plain text.

```bash
# Inspect (admin)
curl -H "Authorization: Bearer $TOKEN" localhost:8000/api/assistant/settings

# Enable for the deployment
curl -X PATCH -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"enabled": true}' localhost:8000/api/assistant/settings

# Sell it as a tier: professional seats only
curl -X PATCH -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"allowed_roles": ["admin", "risk_manager"]}' localhost:8000/api/assistant/settings
```

The UI calls `GET /api/assistant/capabilities` and renders an upsell rather than
a dead feature when the package does not include it.

## Endpoints

| Method | Path | Access |
|---|---|---|
| `GET` | `/api/assistant/capabilities` | any signed-in user |
| `POST` | `/api/assistant` | entitled roles |
| `GET` | `/api/assistant/settings` | admin |
| `PATCH` | `/api/assistant/settings` | admin |

Every question is written to the audit log with the tools it invoked — the
assistant can read customer transaction history, so "who asked what about which
account" has to be answerable.

## Safety notes

- **Read-mostly.** No tool mutates platform state.
- **Declared schemas.** The agent may only call registered tools; arguments are
  coerced and validated, so a model cannot invent a capability.
- **Failures are values.** Tools return `{"error": ...}` rather than raising, so
  one unreachable upstream degrades the answer instead of the request.
- **Bounded.** `max_steps` (1–8) caps tool calls per question.
