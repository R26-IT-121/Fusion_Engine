"""BM25 retrieval over the project corpus.

Deliberately lexical rather than embedding-based. The corpus is ~17 documents
of domain-specific prose where the discriminating terms are exact — "PR-AUC",
"isotonic", "stage 3c", "NOT_APPLICABLE", "0.4056". BM25 matches those directly,
needs no model download, no vector store and no warm-up, and runs in
milliseconds. `sentence-transformers` + Chroma are already project dependencies
if semantic recall is ever needed; `Retriever` is the seam to swap behind.
"""

from __future__ import annotations

import math
from collections import Counter

from chatbot.knowledge import Chunk, tokenize

K1 = 1.5    # term-frequency saturation
B = 0.75    # length normalisation
STALE_PENALTY = 0.55   # down-weight superseded documents, don't hide them
HEADING_BOOST = 1.6    # a term in the section title is a strong topical signal

# A term rarer than this carries real topical meaning; below it we are looking
# at "what", "the", "is". Off-topic questions ("capital of France") match only
# such filler — their distinctive words are absent from the corpus entirely and
# score 0 idf — so a hit that matches no informative term is not an answer.
INFORMATIVE_IDF = 1.5


class Retriever:
    """In-memory BM25 index. Built once at startup, then read-only."""

    def __init__(self, corpus: list[Chunk]):
        self.corpus = corpus
        self.n = len(corpus)
        self.avg_len = (sum(len(c.tokens) for c in corpus) / self.n) if self.n else 0.0
        self.tf: list[Counter] = [Counter(c.tokens) for c in corpus]

        df: Counter = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        # BM25 idf with the +1 guard so common terms never score negative.
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def informative_terms(self, query: str) -> set[str]:
        """Query terms that actually discriminate between documents."""
        return {t for t in tokenize(query) if self.idf.get(t, 0.0) >= INFORMATIVE_IDF}

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        if not terms or not self.n:
            return []

        informative = self.informative_terms(query)
        if not informative:
            # Nothing in the question is specific enough to be about this project.
            return []

        scored: list[tuple[Chunk, float]] = []
        for chunk, counts in zip(self.corpus, self.tf):
            length = len(chunk.tokens) or 1
            score = 0.0
            matched_informative = False
            for term in terms:
                freq = counts.get(term)
                if not freq:
                    continue
                if term in informative:
                    matched_informative = True
                idf = self.idf.get(term, 0.0)
                denom = freq + K1 * (1 - B + B * length / self.avg_len)
                score += idf * (freq * (K1 + 1)) / denom
            # Filler-only matches are noise, not relevance.
            if score <= 0 or not matched_informative:
                continue
            heading_terms = set(tokenize(chunk.heading))
            if heading_terms & informative:
                score *= HEADING_BOOST
            if chunk.stale:
                score *= STALE_PENALTY
            scored.append((chunk, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return self._diversify(scored, top_k)

    @staticmethod
    def _diversify(scored: list[tuple[Chunk, float]], top_k: int) -> list[tuple[Chunk, float]]:
        """Cap passages per document so one long file cannot crowd out the rest."""
        per_source: Counter = Counter()
        out: list[tuple[Chunk, float]] = []
        for chunk, score in scored:
            if per_source[chunk.source] >= 2:
                continue
            per_source[chunk.source] += 1
            out.append((chunk, score))
            if len(out) >= top_k:
                break
        return out
