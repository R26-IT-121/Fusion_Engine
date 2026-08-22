"""
Integration test: Verify DeepSentinel fusion engine can call GraphSAGE API.
Run GraphSAGE first: `python scripts/serve_api.py` from C:\Projects\R26-IT-121\GraphSage

Usage:
    python integration_test_graphsage.py [--graphsage-url http://localhost:8000]
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx


async def test_graphsage_integration(graphsage_url: str = "http://localhost:8000"):
    print(f"\n=== GraphSAGE Integration Test ===")
    print(f"GraphSAGE URL: {graphsage_url}")

    # Sample PaySim transaction
    tx = {
        "transaction_id": "TX_TEST_001",
        "step": 100,
        "type": "TRANSFER",
        "amount": 50000.0,
        "nameOrig": "C123456789",
        "nameDest": "C987654321",
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 50000.0,
        "isFlaggedFraud": 0,
    }

    async with httpx.AsyncClient() as client:
        # 1. Check health
        print("\n1. Checking /health...")
        try:
            health_resp = await client.get(
                f"{graphsage_url}/health", timeout=5.0
            )
            health = health_resp.json()
            print(f"   ✓ GraphSAGE running")
            print(f"   - Stage: {health.get('stage', 'unknown')}")
            print(f"   - Model version: {health.get('model_version', 'unknown')}")
        except Exception as e:
            print(f"   ✗ Health check failed: {e}")
            print(f"   → Is GraphSAGE running? Try: cd C:\\Projects\\R26-IT-121\\GraphSage && python scripts/serve_api.py")
            return False

        # 2. Call /api/graph/analyze
        print("\n2. Calling POST /api/graph/analyze...")
        try:
            score_resp = await client.post(
                f"{graphsage_url}/api/graph/analyze",
                json=tx,
                timeout=10.0,
            )
            print(f"   HTTP {score_resp.status_code}")

            if score_resp.status_code == 200:
                data = score_resp.json()
                score = data.get("relational_risk_score", 0.0)
                risk_level = data.get("risk_level", "UNKNOWN")
                pattern = data.get("suspicious_subgraph", {}).get("pattern", "N/A") if data.get("suspicious_subgraph") else "N/A"

                print(f"   ✓ Scored: {score:.4f} ({risk_level})")
                print(f"   - Pattern: {pattern}")
                print(f"   - Response keys: {list(data.keys())}")

                # 3. Simulate adapter call
                print("\n3. Testing DeepSentinel adapter...")
                try:
                    from backend.adapters.upstream import call_graph_api

                    adapter_resp = await call_graph_api(client, "http://localhost:8000", tx, timeout=10.0)
                    print(f"   ✓ Adapter processed successfully")
                    print(f"   - Score: {adapter_resp.score:.4f}")
                    print(f"   - Available: {adapter_resp.available}")
                    print(f"   - Signal summary: {adapter_resp.fraud_signal_summary}")
                    print(f"   - Typology hint: {adapter_resp.typology_hint}")
                    return True
                except Exception as e:
                    print(f"   ✗ Adapter call failed: {e}")
                    return False

            elif score_resp.status_code == 404:
                print(f"   ! Transaction not in graph (404) — accounts unknown")
                print(f"   (This is normal per contract Section 3)")
                return True

            else:
                print(f"   ✗ Unexpected status: {score_resp.status_code}")
                print(f"   - Body: {score_resp.text[:500]}")
                return False

        except asyncio.TimeoutError:
            print(f"   ✗ Request timeout (10s)")
            print(f"   → GraphSAGE may still be loading (takes ~60-90s on first run)")
            return False
        except Exception as e:
            print(f"   ✗ Request failed: {e}")
            return False


if __name__ == "__main__":
    graphsage_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        graphsage_url = sys.argv[1]

    success = asyncio.run(test_graphsage_integration(graphsage_url))
    sys.exit(0 if success else 1)
