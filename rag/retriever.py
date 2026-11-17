from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path

import chromadb
import numpy as np
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from rag.ingestion import CorpusDocument


class RetrievedChunk(BaseModel):
    id: str
    title: str
    text: str
    score: float
    source: str


class HybridRetriever:
    def __init__(
        self,
        corpus: Sequence[CorpusDocument],
        model_name: str = "all-MiniLM-L6-v2",
        chroma_path: str | Path = "./chroma_db",
        k: int = 10,
    ) -> None:
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not corpus:
            raise ValueError("corpus must contain at least one document")

        self.k = k
        self.model_name = model_name
        self.documents = list(corpus)
        self.documents_by_id = {document.id: document for document in self.documents}
        if len(self.documents_by_id) != len(self.documents):
            raise ValueError("corpus document IDs must be unique")

        self.model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.chroma_client.get_or_create_collection(
            name="financial_rag"
        )

        self.tokenized_documents = [
            self._tokenize(f"{document.title}\n{document.text}")
            for document in self.documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores.astype(float)
        minimum = float(scores.min())
        maximum = float(scores.max())
        if math.isclose(maximum, minimum):
            return (
                np.ones_like(scores, dtype=float)
                if maximum > 0
                else np.zeros_like(scores, dtype=float)
            )
        return (scores - minimum) / (maximum - minimum)

    def retrieve(
        self, query: str, alpha: float = 0.5, return_scores: bool = False
    ) -> list[RetrievedChunk]:
        _ = return_scores
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be a finite number between 0 and 1")
        if not query.strip():
            return []

        dense_scores = self._dense_scores(query) if alpha > 0 else {}
        bm25_scores = self._bm25_scores(query) if alpha < 1 else {}
        candidate_ids = set(dense_scores) | set(bm25_scores)

        combined_scores = {
            document_id: (
                alpha * dense_scores.get(document_id, 0.0)
                + (1 - alpha) * bm25_scores.get(document_id, 0.0)
            )
            for document_id in candidate_ids
        }
        ranked_ids = sorted(
            candidate_ids,
            key=lambda document_id: (
                combined_scores[document_id],
                document_id,
            ),
            reverse=True,
        )[: self.k]

        results: list[RetrievedChunk] = []
        for document_id in ranked_ids:
            document = self.documents_by_id[document_id]
            has_dense_score = document_id in dense_scores and alpha > 0
            has_bm25_score = document_id in bm25_scores and alpha < 1
            source = (
                "hybrid"
                if has_dense_score and has_bm25_score
                else "chroma"
                if has_dense_score
                else "bm25"
            )
            results.append(
                RetrievedChunk(
                    id=document.id,
                    title=document.title,
                    text=document.text,
                    score=combined_scores[document_id],
                    source=source,
                )
            )
        return results

    def _dense_scores(self, query: str) -> dict[str, float]:
        query_embedding = np.asarray(
            self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        ).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(self.k, len(self.documents)),
            include=["distances"],
        )
        ids = results["ids"][0]
        distances = np.asarray(results["distances"][0], dtype=float)
        similarities = 1.0 / (1.0 + distances)
        normalized = self._normalize_scores(similarities)
        return dict(zip(ids, normalized.tolist(), strict=True))

    def _bm25_scores(self, query: str) -> dict[str, float]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {}

        scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=float)
        matching_indices = [
            index
            for index in np.argsort(scores)[::-1]
            if any(token in self.tokenized_documents[index] for token in query_tokens)
        ][: self.k]
        if not matching_indices:
            return {}

        matching_scores = scores[matching_indices]
        normalized = self._normalize_scores(matching_scores)
        return {
            self.documents[index].id: float(score)
            for index, score in zip(matching_indices, normalized, strict=True)
        }


def retrieve_hybrid(
    corpus: Sequence[CorpusDocument],
    query: str,
    k: int = 10,
    alpha: float = 0.5,
    model_name: str = "all-MiniLM-L6-v2",
    chroma_path: str | Path = "./chroma_db",
) -> list[RetrievedChunk]:
    retriever = HybridRetriever(
        corpus=corpus,
        model_name=model_name,
        chroma_path=chroma_path,
        k=k,
    )
    return retriever.retrieve(query=query, alpha=alpha)


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    from rag.ingestion import load_corpus, load_queries

    corpus = load_corpus(base_path / "data" / "corpus.jsonl")
    queries = load_queries(base_path / "data" / "queries.jsonl")
    retriever = HybridRetriever(corpus=corpus, k=5)

    if queries:
        for result in retriever.retrieve(queries[0].text):
            print(f"{result.score:.3f} [{result.source}] {result.title}")
            print(f"  {result.text[:200]}...")
