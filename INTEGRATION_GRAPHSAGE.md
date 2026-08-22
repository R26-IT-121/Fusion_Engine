# Integration with GraphSAGE (Member 2 — Ewaduge)

## Quick Start

### 1. Start the GraphSAGE API

```powershell
cd C:\Projects\R26-IT-121\GraphSage
python -m pip install -e .
python scripts/serve_api.py
```

The API will start on `http://localhost:8000` and take ~60-90 seconds to load the model.

Check health:
```bash
curl http://localhost:8000/health
```

### 2. Update DeepSentinel `.env`

The `.env` is already set up to call GraphSAGE at `http://localhost:8000`:
```ini
GRAPH_API_BASE=http://localhost:8000
BEHAVIORAL_API_BASE=http://localhost:8001
TEMPORAL_API_BASE=http://localhost:8003
```

### 3. Test the integration

```bash
cd C:\Projects\DeepSentinel
python integration_test_graphsage.py
```

This will:
- Check GraphSAGE health
- Send a test transaction to `/api/graph/analyze`
- Verify the adapter correctly extracts the response
- Report any errors

### 4. Send a full transaction through fusion engine

Once GraphSAGE is running and the integration test passes:

```bash
python -c "
import asyncio
from backend.main import app, analyze
from pydantic import BaseModel

async def test():
    from fastapi.testclient import TestClient
    from backend.main import AnalyzeRequest, TransactionData

    req = AnalyzeRequest(
        transaction=TransactionData(
            step=100,
            type='TRANSFER',
            amount=50000.0,
            nameOrig='C123456789',
            nameDest='C987654321',
            oldbalanceOrg=50000.0,
            newbalanceOrig=0.0,
            oldbalanceDest=0.0,
            newbalanceDest=50000.0,
            isFlaggedFraud=0,
        )
    )
    result = await analyze(req)
    print(result.model_dump_json(indent=2))

asyncio.run(test())
"
```

## What Happens

1. **Fusion engine receives transaction** → calls GraphSAGE API (+ behavioral, temporal when ready)
2. **GraphSAGE adapter** (in `backend/adapters/upstream.py:call_graph_api`) handles:
   - Building the request (`POST /api/graph/analyze`)
   - Extracting the response (`relational_risk_score`, `suspicious_subgraph`)
   - Handling error cases (404 = account unknown, NOT_APPLICABLE = type mismatch)
   - Building a human-readable `fraud_signal_summary` from pattern + sink + convergence count
3. **Response flows to meta-classifier** → fused with other modality scores
4. **RAG + LLM** → uses the `fraud_signal_summary` as grounded evidence

## API Contract

| Property | Value |
|----------|-------|
| Endpoint | `POST /api/graph/analyze` |
| Request | Full PaySim transaction (see `integration_test_graphsage.py`) |
| Response | `relational_risk_score` (0–1), `suspicious_subgraph` with pattern/nodes/edges |
| Errors | 404 (unknown accounts), NOT_APPLICABLE (non-TRANSFER/CASH_OUT type) |
| Latency | p95 < 500 ms (no inference per request, pre-computed scores) |

**Full contract:** [`C:\Projects\R26-IT-121\GraphSage\docs\integration\graph_api_contract.md`](../../../R26-IT-121/GraphSage/docs/integration/graph_api_contract.md)

## Troubleshooting

**"Connection refused" / "Unable to connect to http://localhost:8000"**
→ GraphSAGE not running. Check terminal for errors during `python scripts/serve_api.py`. Model loads take ~60-90 seconds.

**"404 Not Found" from integration test**
→ This is normal per contract. Means the transaction's accounts are not in the graph (fixed snapshot).

**Timeout after 5–10 seconds**
→ GraphSAGE still loading. Wait another minute and retry.

**422 Validation error**
→ Transaction fields malformed. Check `integration_test_graphsage.py` for the correct schema.

## Next Steps

- M1 (Wijesinghe) VAE/DSAA API — integrate the same way (endpoint: `/api/v1/behavioral/classify`)
- M3 (Pathirana) TCN/TSCFD API — integrate the same way (endpoint: `/api/v1/classify`)
- Once all three upstream APIs are running, the full pipeline is live
