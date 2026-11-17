from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag.ingestion import CorpusDocument, Query, load_corpus, load_queries
from rag.retriever import HybridRetriever


@dataclass(frozen=True)
class RetrievalMetrics:
    recall: float
    mrr: float
    query_count: int


def load_qrels(path: Path | str) -> dict[str, set[str]]:
    file_path = Path(path)
    relevance: dict[str, set[str]] = {}
    with file_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        expected_fields = {"query_id", "corpus_id", "score"}
        if reader.fieldnames is None or not expected_fields.issubset(reader.fieldnames):
            raise ValueError(
                f"{file_path} must contain tab-separated columns "
                "query_id, corpus_id, and score"
            )
        for line_number, row in enumerate(reader, start=2):
            query_id = (row.get("query_id") or "").strip()
            corpus_id = (row.get("corpus_id") or "").strip()
            raw_score = (row.get("score") or "").strip()
            if not query_id or not corpus_id or not raw_score:
                raise ValueError(
                    f"Missing qrels value in {file_path} at line {line_number}"
                )
            try:
                score = float(raw_score)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid relevance score {raw_score!r} in {file_path} "
                    f"at line {line_number}"
                ) from exc
            if not math.isfinite(score) or score < 0:
                raise ValueError(
                    f"Relevance score must be finite and non-negative in "
                    f"{file_path} at line {line_number}"
                )
            if score > 0:
                relevance.setdefault(query_id, set()).add(corpus_id)
    if not relevance:
        raise ValueError(f"No positive relevance judgments found in {file_path}")
    return relevance


def _measure(
    retrieved_ids: Sequence[str], relevant_ids: set[str]
) -> tuple[float, float]:
    if not relevant_ids:
        return 0.0, 0.0
    hits = [
        rank
        for rank, doc_id in enumerate(retrieved_ids, start=1)
        if doc_id in relevant_ids
    ]
    recall = len({retrieved_ids[rank - 1] for rank in hits}) / len(relevant_ids)
    reciprocal_rank = 1.0 / hits[0] if hits else 0.0
    return recall, reciprocal_rank


def evaluate_retrieval(
    corpus: Sequence[CorpusDocument],
    queries: Sequence[Query],
    qrels_path: Path | str,
    k: int = 10,
    model_name: str = "all-MiniLM-L6-v2",
    chroma_path: str | Path = "./chroma_db",
) -> dict[str, RetrievalMetrics]:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not corpus:
        raise ValueError("corpus must contain at least one document")

    qrels = load_qrels(qrels_path)
    queries_by_id = {query.id: query for query in queries}
    missing_queries = qrels.keys() - queries_by_id.keys()
    if missing_queries:
        raise ValueError(
            "Queries referenced by qrels are missing from the query data: "
            + ", ".join(sorted(missing_queries))
        )
    corpus_ids = {document.id for document in corpus}
    missing_documents = {
        document_id
        for relevant_ids in qrels.values()
        for document_id in relevant_ids
        if document_id not in corpus_ids
    }
    if missing_documents:
        raise ValueError(
            "Relevant documents referenced by qrels are missing from the corpus: "
            + ", ".join(sorted(missing_documents))
        )

    retriever = HybridRetriever(
        corpus=corpus,
        model_name=model_name,
        chroma_path=chroma_path,
        k=k,
    )
    return evaluate_with_retriever(retriever, queries, qrels, k=k)


def evaluate_with_retriever(
    retriever: HybridRetriever,
    queries: Sequence[Query],
    qrels: Mapping[str, set[str]],
    k: int | None = None,
) -> dict[str, RetrievalMetrics]:
    cutoff = retriever.k if k is None else k
    if not isinstance(cutoff, int) or isinstance(cutoff, bool) or cutoff <= 0:
        raise ValueError("k must be a positive integer")

    queries_by_id = {query.id: query for query in queries}
    missing_queries = qrels.keys() - queries_by_id.keys()
    if missing_queries:
        raise ValueError(
            "Queries referenced by qrels are missing from the query data: "
            + ", ".join(sorted(missing_queries))
        )

    methods = {
        "BM25-only": 0.0,
        "Dense-only": 1.0,
        "Hybrid": 0.5,
    }
    totals = {name: [0.0, 0.0] for name in methods}
    for query_id, relevant_ids in qrels.items():
        query = queries_by_id[query_id]
        for method, alpha in methods.items():
            results = retriever.retrieve(query.text, alpha=alpha)
            recall, reciprocal_rank = _measure(
                [result.id for result in results[:cutoff]],
                relevant_ids,
            )
            totals[method][0] += recall
            totals[method][1] += reciprocal_rank

    query_count = len(qrels)
    if not query_count:
        raise ValueError("At least one query must have positive qrels judgments")
    return {
        method: RetrievalMetrics(
            recall=recall_total / query_count,
            mrr=mrr_total / query_count,
            query_count=query_count,
        )
        for method, (recall_total, mrr_total) in totals.items()
    }


def main() -> None:
    base_path = Path(__file__).parent.parent
    corpus = load_corpus(base_path / "data" / "corpus.jsonl")
    queries = load_queries(base_path / "data" / "queries.jsonl")
    metrics = evaluate_retrieval(
        corpus=corpus,
        queries=queries,
        qrels_path=base_path / "data" / "FinanceBench_qrels.tsv",
        chroma_path=base_path / "chroma_db",
    )
    print(
        f"FinanceBench retrieval metrics (k=10, queries={next(iter(metrics.values())).query_count})"
    )
    print(f"{'Method':<14} {'Recall@10':>12} {'MRR@10':>10}")
    for method, result in metrics.items():
        print(f"{method:<14} {result.recall:>12.4f} {result.mrr:>10.4f}")


if __name__ == "__main__":
    main()
