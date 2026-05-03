"""
FATF Fraud Typology Knowledge Base
Loads FATF typologies from JSON, embeds them with sentence-transformers,
and stores them in a local ChromaDB collection for semantic retrieval.
"""

import json
import os
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

COLLECTION_NAME = "fatf_typologies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class FATFKnowledgeBase:
    def __init__(self, chroma_db_path: str, fatf_data_path: str):
        self.chroma_db_path = chroma_db_path
        self.fatf_data_path = fatf_data_path
        self._client: chromadb.PersistentClient | None = None
        self._collection = None
        self._embedder: SentenceTransformer | None = None

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    def initialize(self):
        """Build or load the ChromaDB collection from FATF typologies."""
        os.makedirs(self.chroma_db_path, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self.chroma_db_path,
            settings=Settings(anonymized_telemetry=False),
        )

        existing = [c.name for c in self._client.list_collections()]

        if COLLECTION_NAME in existing:
            self._collection = self._client.get_collection(COLLECTION_NAME)
            logger.info(
                f"Loaded existing ChromaDB collection '{COLLECTION_NAME}' "
                f"with {self._collection.count()} typologies."
            )
            return

        logger.info("Building new ChromaDB collection from FATF typologies...")
        typologies = self._load_typologies()
        self._collection = self._client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._ingest(typologies)
        logger.info(
            f"Ingested {self._collection.count()} FATF typologies into ChromaDB."
        )

    def _load_typologies(self) -> list[dict]:
        path = Path(self.fatf_data_path)
        if not path.exists():
            raise FileNotFoundError(f"FATF data file not found: {path.resolve()}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_document(self, typology: dict) -> str:
        """Combine typology fields into a single rich text for embedding."""
        indicators = "\n".join(f"- {i}" for i in typology.get("indicators", []))
        return (
            f"Typology: {typology['name']}\n"
            f"Stage: {typology.get('stage', 'Unknown')}\n"
            f"Risk Level: {typology.get('risk_level', 'Unknown')}\n"
            f"Description: {typology['description']}\n"
            f"Behavioral Indicators:\n{indicators}\n"
            f"Typical Patterns: {typology.get('typical_transaction_patterns', '')}\n"
            f"Graph Signal: {typology.get('graph_signal', 'unknown')} | "
            f"Behavioral Signal: {typology.get('behavioral_signal', 'unknown')} | "
            f"Temporal Signal: {typology.get('temporal_signal', 'unknown')}"
        )

    def _ingest(self, typologies: list[dict]):
        embedder = self._get_embedder()
        documents = [self._build_document(t) for t in typologies]
        embeddings = embedder.encode(documents, show_progress_bar=True).tolist()

        self._collection.add(
            ids=[t["id"] for t in typologies],
            documents=documents,
            embeddings=embeddings,
            metadatas=[
                {
                    "name": t["name"],
                    "stage": t.get("stage", ""),
                    "risk_level": t.get("risk_level", ""),
                    "graph_signal": t.get("graph_signal", ""),
                    "behavioral_signal": t.get("behavioral_signal", ""),
                    "temporal_signal": t.get("temporal_signal", ""),
                }
                for t in typologies
            ],
        )

    def get_collection(self):
        if self._collection is None:
            raise RuntimeError("Knowledge base not initialized. Call initialize() first.")
        return self._collection

    def get_embedder(self) -> SentenceTransformer:
        return self._get_embedder()

    def rebuild(self):
        """Force a full rebuild of the collection (e.g., after updating typologies)."""
        if self._client and COLLECTION_NAME in [c.name for c in self._client.list_collections()]:
            self._client.delete_collection(COLLECTION_NAME)
            logger.info("Deleted existing collection for rebuild.")
        self._collection = None
        self.initialize()
