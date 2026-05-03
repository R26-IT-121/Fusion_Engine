"""
Quick smoke test for the RAG pipeline (no LLM / no Gemini API key needed).
Run from the C:/Projects/DeepSentinel directory:
    python test_pipeline.py
"""

import os
import sys
import logging

# Ensure backend package is importable
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("test")

CHROMA_DB_PATH = "./chroma_store_test"
FATF_DATA_PATH = "./data/fatf_typologies.json"
MODEL_SAVE_PATH = "./models/meta_classifier.joblib"


def test_knowledge_base():
    logger.info("=== TEST 1: FATF Knowledge Base ===")
    from backend.rag.knowledge_base import FATFKnowledgeBase

    kb = FATFKnowledgeBase(
        chroma_db_path=CHROMA_DB_PATH,
        fatf_data_path=FATF_DATA_PATH,
    )
    kb.initialize()
    collection = kb.get_collection()
    count = collection.count()
    assert count == 10, f"Expected 10 typologies, got {count}"
    logger.info(f"PASS — {count} typologies loaded into ChromaDB.")
    return kb


def test_retriever(kb):
    logger.info("=== TEST 2: RAG Retriever ===")
    from backend.rag.retriever import FATFRetriever

    retriever = FATFRetriever(
        collection=kb.get_collection(),
        embedder=kb.get_embedder(),
        top_k=1,
    )

    # Layering-like profile: high graph + high behavioral + high temporal
    results = retriever.retrieve(
        graph_score=0.88,
        behavioral_score=0.82,
        temporal_score=0.79,
        confidence_score=0.85,
    )
    assert results, "No retrieval results returned."
    top = results[0]
    logger.info(f"PASS — Top match: '{top.typology_name}' (similarity={top.similarity_score:.2%})")

    # Account Takeover: low graph, high behavioral, high temporal
    results2 = retriever.retrieve(
        graph_score=0.15,
        behavioral_score=0.91,
        temporal_score=0.84,
        confidence_score=0.76,
    )
    top2 = results2[0]
    logger.info(f"PASS — ATO profile match: '{top2.typology_name}' (similarity={top2.similarity_score:.2%})")

    return retriever


def test_prompt_builder(retriever):
    logger.info("=== TEST 3: Chain of Evidence Prompt Builder ===")
    from backend.rag.retriever import FATFRetriever, RetrievalResult
    from backend.rag.prompt_builder import build_chain_of_evidence_prompt

    mock_retrieval = retriever.retrieve(
        graph_score=0.88, behavioral_score=0.82,
        temporal_score=0.79, confidence_score=0.85
    )[0]

    package = build_chain_of_evidence_prompt(
        transaction_id="TXN-TEST-001",
        graph_score=0.88,
        behavioral_score=0.82,
        temporal_score=0.79,
        confidence_score=0.85,
        graph_available=True,
        behavioral_available=True,
        temporal_available=True,
        retrieval=mock_retrieval,
    )
    assert "chain of evidence" in package.system_prompt.lower()
    assert "TXN-TEST-001" in package.user_prompt
    assert mock_retrieval.typology_id in package.user_prompt
    logger.info(f"PASS — Prompt built. System prompt: {len(package.system_prompt)} chars, User prompt: {len(package.user_prompt)} chars.")
    return package


def test_meta_classifier():
    logger.info("=== TEST 4: Meta Classifier (Logistic Regression) ===")
    from backend.fusion_engine import MetaClassifier

    clf = MetaClassifier(model_save_path=MODEL_SAVE_PATH)
    clf.initialize()

    # High fraud: all scores high
    result = clf.fuse(graph_score=0.88, behavioral_score=0.82, temporal_score=0.79)
    logger.info(f"High fraud input → confidence: {result.confidence_score:.4f}")
    assert result.confidence_score > 0.5, "Expected high confidence for high-fraud inputs."

    # Low fraud: all scores low
    result2 = clf.fuse(graph_score=0.05, behavioral_score=0.08, temporal_score=0.03)
    logger.info(f"Low fraud input  → confidence: {result2.confidence_score:.4f}")
    assert result2.confidence_score < 0.5, "Expected low confidence for legitimate inputs."

    # Missing modality
    result3 = clf.fuse(graph_score=0.88, behavioral_score=None, temporal_score=0.79)
    logger.info(f"Missing behavioral → confidence: {result3.confidence_score:.4f}, modalities_used={result3.modalities_used}")
    assert result3.modalities_used == 2
    logger.info("PASS — Meta-classifier producing sensible outputs.")


def test_mock_scores():
    logger.info("=== TEST 5: Mock Score Generator ===")
    from backend.mock_scores import generate_mock_scores, FraudScenario

    for scenario in [
        FraudScenario.SMURFING,
        FraudScenario.LAYERING,
        FraudScenario.ACCOUNT_TAKEOVER,
        FraudScenario.LEGITIMATE,
    ]:
        scores = generate_mock_scores(scenario=scenario, seed=42)
        logger.info(
            f"  {scenario.value:20s} → G={scores.graph_score:.3f} "
            f"B={scores.behavioral_score:.3f} T={scores.temporal_score:.3f}"
        )
    logger.info("PASS — Mock scores generated for all scenarios.")


if __name__ == "__main__":
    logger.info("Starting DeepSentinel RAG Pipeline smoke test...")
    try:
        kb = test_knowledge_base()
        retriever = test_retriever(kb)
        test_prompt_builder(retriever)
        test_meta_classifier()
        test_mock_scores()
        logger.info("\n=== ALL TESTS PASSED ===")
        logger.info("The RAG pipeline (minus LLM) is fully operational.")
        logger.info("Next step: add your GEMINI_API_KEY to .env and run the FastAPI server.")
    except Exception as e:
        logger.error(f"TEST FAILED: {e}", exc_info=True)
        sys.exit(1)
