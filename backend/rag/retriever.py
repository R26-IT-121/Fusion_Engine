"""
FATF Typology Retriever
Converts a multi-modal risk profile into a semantic search query,
then performs cosine similarity search against the ChromaDB knowledge base.
"""

import logging
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    typology_id: str
    typology_name: str
    stage: str
    risk_level: str
    document: str
    similarity_score: float  # 0.0 (no match) → 1.0 (perfect match)
    metadata: dict


class FATFRetriever:
    def __init__(self, collection, embedder: SentenceTransformer, top_k: int = 1):
        self._collection = collection
        self._embedder = embedder
        self.top_k = top_k

    # What each piece of detector evidence means, in the vocabulary the FATF
    # typology documents use. Detector labels on their own ("HUB_AND_SPOKE")
    # do not embed near the typology text; these descriptions do.
    PATTERN_TEXT = {
        "HUB_AND_SPOKE": "Many accounts send money to one collecting mule account, which forwards or cashes it out.",
        "SMURFING": "A large sum is split into many small transfers kept below reporting thresholds.",
        "LAYERING": "Funds pass quickly through a chain of intermediary accounts to hide their origin.",
        "ACCOUNT_TAKEOVER": "A legitimate account is taken over and its balance is moved out.",
    }

    # A structural pattern the graph detector found is stronger evidence of the
    # typology than any wording: it was measured across several accounts, which
    # a single transaction's facts cannot show. It decides the ranking among
    # the closest candidates; the similarity reported stays the real one.
    PATTERN_TYPOLOGY = {
        "HUB_AND_SPOKE": "FATF-003",
        "SMURFING": "FATF-001",
        "LAYERING": "FATF-002",
        "ACCOUNT_TAKEOVER": "FATF-004",
    }
    PATTERN_BOOST = 0.15

    @staticmethod
    def _level(score: float) -> str:
        if score >= 0.9:
            return "very strong"
        if score >= 0.75:
            return "strong"
        if score >= 0.5:
            return "moderate"
        if score >= 0.25:
            return "weak"
        return "no"

    def _build_query(
        self,
        graph_score: float,
        behavioral_score: float,
        temporal_score: float,
        confidence_score: float,
        evidence: dict | None = None,
    ) -> str:
        """
        Describe the case in words the typology documents use.

        The first version described only three score levels, so every alert
        produced nearly the same query: across the 442 alerts of the replayed
        test window it matched just two of the ten typologies. Detector evidence
        and transaction facts are what tell one fraud pattern from another, so
        they lead the query when they are available; the scores follow.
        """
        ev = evidence or {}
        parts = []

        pattern = str(ev.get("graph_pattern") or "").upper()
        if pattern in self.PATTERN_TEXT:
            parts.append(self.PATTERN_TEXT[pattern])
        if ev.get("behaviour_hint"):
            parts.append(f"Behavioural anomaly resembles {ev['behaviour_hint']}.")

        tx_type = str(ev.get("type") or "").upper()
        if ev.get("sender_drained"):
            parts.append("The sending account is emptied in a single transaction, typical of account takeover.")
        if ev.get("receiver_was_empty"):
            parts.append("The receiving account had no prior balance, typical of a new mule account.")
        if tx_type == "CASH_OUT":
            parts.append("The funds are cashed out.")
        elif tx_type == "TRANSFER":
            parts.append("The funds are transferred to another account.")
        amount = ev.get("amount")
        if isinstance(amount, (int, float)) and amount > 0:
            if amount < 10_000:
                parts.append("The amount is small, consistent with structuring into many small payments.")
            elif amount >= 1_000_000:
                parts.append("The amount is very large.")

        g, b, t = (self._level(x) for x in (graph_score, behavioral_score, temporal_score))
        parts.append(
            f"Network evidence is {g}, behavioural anomaly is {b}, and rapid sequential activity is {t}. "
            f"Overall fraud confidence is {confidence_score:.0%}."
        )
        if graph_score >= 0.75 and temporal_score >= 0.75:
            parts.append("Multiple accounts move money in rapid succession.")
        if behavioral_score >= 0.75 and graph_score < 0.25:
            parts.append("A sudden departure from the account's normal behaviour without network involvement.")
        if temporal_score >= 0.75 and behavioral_score < 0.25:
            parts.append("An automated high-velocity burst of transactions.")
        return " ".join(parts)

    def retrieve(
        self,
        graph_score: float,
        behavioral_score: float,
        temporal_score: float,
        confidence_score: float,
        evidence: dict | None = None,
    ) -> list[RetrievalResult]:
        query = self._build_query(
            graph_score, behavioral_score, temporal_score, confidence_score, evidence
        )
        logger.info(f"RAG query: {query}")

        query_embedding = self._embedder.encode([query]).tolist()

        # Rank a wider pool so detector evidence can lift the right typology
        # even when the wording alone puts it a place or two lower.
        pool = max(self.top_k, min(10, self._collection.count()))
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=pool,
            include=["documents", "metadatas", "distances"],
        )
        favoured = self.PATTERN_TYPOLOGY.get(str((evidence or {}).get("graph_pattern") or "").upper())

        retrieval_results = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # ChromaDB cosine distance = 1 - cosine_similarity
            similarity = 1.0 - distance

            retrieval_results.append(
                RetrievalResult(
                    typology_id=results["ids"][0][i],
                    typology_name=results["metadatas"][0][i]["name"],
                    stage=results["metadatas"][0][i]["stage"],
                    risk_level=results["metadatas"][0][i]["risk_level"],
                    document=results["documents"][0][i],
                    similarity_score=round(similarity, 4),
                    metadata=results["metadatas"][0][i],
                )
            )

        retrieval_results.sort(
            key=lambda r: r.similarity_score + (self.PATTERN_BOOST if r.typology_id == favoured else 0.0),
            reverse=True,
        )
        retrieval_results = retrieval_results[: self.top_k]

        top = retrieval_results[0] if retrieval_results else None
        if top:
            logger.info(
                f"Top match: '{top.typology_name}' "
                f"(similarity={top.similarity_score:.2%})"
            )

        return retrieval_results
