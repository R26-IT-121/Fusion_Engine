# DeepSentinel

A multi-modal fraud detection and explainability engine that fuses signals from three upstream analytical models (Graph Neural Network, Behavioral VAE, Temporal CNN) and generates forensic reports grounded in FATF regulatory typologies via RAG.

---

## Architecture

```
Transaction / Upstream Scores
         ↓
  [Fusion Engine]  ←  Graph score + Behavioral score + Temporal score
  (Logistic Regression meta-classifier)
         ↓
  [FATF Retriever]  ←  ChromaDB cosine similarity search
  (Top-1 typology match)
         ↓
  [Forensic Reporter]  ←  Chain-of-Evidence prompt + retrieved typology
  (Gemini / Ollama LLM)
         ↓
  AnalyzeResponse (confidence, classification, retrieval metadata, report)
```

---

## Components

| Component | File | Description |
|-----------|------|-------------|
| API Server | `backend/main.py` | FastAPI app; orchestrates the full pipeline |
| Fusion Engine | `backend/fusion_engine.py` | Weighted meta-classifier; fuses 3 modality scores |
| FATF Knowledge Base | `backend/rag/knowledge_base.py` | ChromaDB + sentence-transformers embeddings |
| Semantic Retriever | `backend/rag/retriever.py` | Converts score profile to semantic query; retrieves typology |
| Prompt Builder | `backend/rag/prompt_builder.py` | Chain-of-Evidence and baseline prompts |
| Forensic Reporter | `backend/llm/forensic_reporter.py` | LLM backend abstraction (Gemini / Ollama) |
| Mock Generator | `backend/mock_scores.py` | Synthetic upstream scores for 6 fraud scenarios |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health and initialization status |
| `GET` | `/typologies` | List all FATF typologies in the knowledge base |
| `POST` | `/analyze` | Analyze a transaction using scores or mock data |
| `POST` | `/analyze/transaction` | Analyze a full PaySim-style transaction record |
| `POST` | `/retrain` | Force retrain the meta-classifier |
| `POST` | `/rebuild-kb` | Rebuild ChromaDB from the FATF typology JSON |

### `POST /analyze` — Request

```json
{
  "transaction_id": "string (auto-generated if omitted)",
  "graph_score": 0.0,
  "behavioral_score": 0.0,
  "temporal_score": 0.0,
  "use_mock": false,
  "mock_scenario": "smurfing | layering | mule_network | account_takeover | velocity_fraud | legitimate",
  "include_baseline": false
}
```

### `POST /analyze` — Response

```json
{
  "transaction_id": "string",
  "fraud_confidence_score": 0.0,
  "classification": "CRITICAL | HIGH | MEDIUM | LOW",
  "graph_score": 0.0,
  "behavioral_score": 0.0,
  "temporal_score": 0.0,
  "graph_available": true,
  "behavioral_available": true,
  "temporal_available": true,
  "modalities_used": 3,
  "retrieval": {
    "typology_id": "string",
    "name": "string",
    "stage": "string",
    "risk_level": "string",
    "similarity_score": 0.0
  },
  "forensic_report": "string",
  "baseline_report": "string (if include_baseline=true)",
  "mock_scenario": "string (if mocks used)"
}
```

**Classification thresholds:**

| Score | Label |
|-------|-------|
| ≥ 0.80 | CRITICAL |
| 0.65 – 0.79 | HIGH |
| 0.50 – 0.64 | MEDIUM |
| < 0.50 | LOW |

---

## Setup

### Prerequisites

- Python 3.10+
- A Gemini API key **or** a local [Ollama](https://ollama.com) instance

### Install

```bash
git clone https://github.com/your-org/DeepSentinel.git
cd DeepSentinel
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
LLM_PROVIDER=gemini          # or: ollama
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash

CHROMA_DB_PATH=./chroma_store
FATF_DATA_PATH=./data/fatf_typologies.json
MODEL_SAVE_PATH=./models/meta_classifier.joblib

# Upstream model endpoints (update when teammates deploy)
GRAPH_MODEL_URL=http://localhost:8001/predict
BEHAVIORAL_MODEL_URL=http://localhost:8002/predict
TEMPORAL_MODEL_URL=http://localhost:8003/predict
UPSTREAM_TIMEOUT_MS=500
```

### Run

```bash
uvicorn backend.main:app --reload
```

The service starts at `http://localhost:8000`. The knowledge base is built and the meta-classifier is trained automatically on first start.

### Test

```bash
python test_pipeline.py
```

---

## Upstream Model Integration

DeepSentinel calls three external model APIs in parallel. Each must return a JSON body with a `fraud_score` field (float 0–1):

```json
{ "fraud_score": 0.87 }
```

If any upstream call times out (default 500 ms) or fails, DeepSentinel continues with the available modalities and applies a 10% confidence penalty per missing modality.

### Expected score keys per teammate API

| Model | Owner | Expected key |
|-------|-------|-------------|
| Graph (GraphSAGE) | Member 1 | `fraud_score` |
| Behavioral (VAE) | Member 2 | `fraud_score` |
| Temporal (CNN) | Member 3 | `fraud_score` |

---

## Mock Scenarios

Use `use_mock: true` with any `mock_scenario` value to bypass upstream calls:

| Scenario | Graph | Behavioral | Temporal |
|----------|-------|------------|---------|
| `smurfing` | medium | medium | high (0.75–0.92) |
| `layering` | high | high | high (0.80–0.95) |
| `mule_network` | high | high | medium |
| `account_takeover` | low | high | high |
| `velocity_fraud` | low | medium | very high (0.88–0.99) |
| `legitimate` | low | low | low (0.02–0.20) |

---

## RAG Knowledge Base

10 FATF fraud typologies are stored in `data/fatf_typologies.json` and embedded using `sentence-transformers/all-MiniLM-L6-v2` (384-dim vectors) in a ChromaDB persistent store.

The retriever converts the incoming score profile into a natural-language query (e.g., *"high graph network signal, medium behavioral anomaly signal"*) and returns the closest typology by cosine similarity.

The retrieved typology is injected into a **Chain-of-Evidence prompt** that instructs the LLM to cite only the provided numerical evidence and regulatory text — no hallucinated context.

Setting `include_baseline: true` also generates an ungrounded report (scores only, no typology) for ablation comparison.

---

## Deployment

The service is configured for [Render.com](https://render.com) via `render.yaml`:

```bash
# Build command
pip install -r requirements.txt

# Start command
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Set all `.env` variables as environment variables in the Render dashboard. Update `GRAPH_MODEL_URL`, `BEHAVIORAL_MODEL_URL`, and `TEMPORAL_MODEL_URL` to point to your teammates' deployed services.

---

## Project Structure

```
DeepSentinel/
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── fusion_engine.py        # Meta-classifier
│   ├── mock_scores.py          # Mock upstream scores
│   ├── llm/
│   │   └── forensic_reporter.py
│   └── rag/
│       ├── knowledge_base.py
│       ├── retriever.py
│       └── prompt_builder.py
├── data/
│   └── fatf_typologies.json    # 10 FATF typologies
├── models/                     # Auto-created; trained classifier
├── chroma_store/               # Auto-created; ChromaDB vector store
├── test_pipeline.py
├── requirements.txt
├── render.yaml
├── Procfile
└── .env.example
```

---

## Team

| Member | Component | Stack |
|--------|-----------|-------|
| Member 1 | Graph model (GraphSAGE) | PyTorch Geometric |
| Member 2 | Behavioral model (VAE) | PyTorch |
| Member 3 | Temporal model (CNN) | TensorFlow |
| Member 4 | Fusion engine + explainability (this repo) | FastAPI, scikit-learn, ChromaDB, Gemini |
