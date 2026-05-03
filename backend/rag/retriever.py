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

    def _build_query(
        self,
        graph_score: float,
        behavioral_score: float,
        temporal_score: float,
        confidence_score: float,
    ) -> str:
        """
        Translate numerical risk scores into a natural language query that
        captures the qualitative profile of the fraud signal.
        This bridges the numerical ML output to the semantic embedding space.
        """

        def level(score: float) -> str:
            if score >= 0.75:
                return "high"
            if score >= 0.40:
                return "medium"
            return "low"

        graph_lvl = level(graph_score)
        behavioral_lvl = level(behavioral_score)
        temporal_lvl = level(temporal_score)

        dominant = max(
            [("graph", graph_score), ("behavioral", behavioral_score), ("temporal", temporal_score)],
            key=lambda x: x[1],
        )

        query = (
            f"Financial fraud with {graph_lvl} graph network signal, "
            f"{behavioral_lvl} behavioral anomaly signal, and {temporal_lvl} temporal pattern signal. "
            f"Overall fraud confidence is {confidence_score:.0%}. "
            f"Dominant detection modality is {dominant[0]} analysis. "
        )

        if graph_lvl == "high" and temporal_lvl == "high":
            query += "Involves multiple accounts with rapid sequential transfers. "
        if behavioral_lvl == "high" and graph_lvl == "low":
            query += "Sudden deviation from account baseline without network involvement. "
        if graph_lvl == "high" and behavioral_lvl == "high":
            query += "Complex multi-account network with abnormal user behavior. "
        if temporal_lvl == "high" and behavioral_lvl == "low":
            query += "Automated high-velocity transaction pattern. "

        return query.strip()

    def retrieve(
        self,
        graph_score: float,
        behavioral_score: float,
        temporal_score: float,
        confidence_score: float,
    ) -> list[RetrievalResult]:
        query = self._build_query(
            graph_score, behavioral_score, temporal_score, confidence_score
        )
        logger.info(f"RAG query: {query}")

        query_embedding = self._embedder.encode([query]).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

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

        top = retrieval_results[0] if retrieval_results else None
        if top:
            logger.info(
                f"Top match: '{top.typology_name}' "
                f"(similarity={top.similarity_score:.2%})"
            )

        return retrieval_results
