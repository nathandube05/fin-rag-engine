"""Tests for the financial RAG engine that need NO API key and NO network."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.ingestion import load_corpus, load_queries


def test_load_corpus_parses_data_files():
    """load_corpus parses the corpus JSONL file correctly."""
    base_path = Path(__file__).parent.parent
    corpus = load_corpus(base_path / "data" / "corpus.jsonl")
    assert len(corpus) > 0, "Corpus should contain documents"
    # Check a sample document has expected fields
    doc = corpus[0]
    assert hasattr(doc, "id"), "Document should have an id"
    assert hasattr(doc, "title"), "Document should have a title"
    assert hasattr(doc, "text"), "Document should have text"
    print(f"  Loaded {len(corpus)} corpus documents")


def test_load_queries_parses_data_files():
    """load_queries parses the queries JSONL file correctly."""
    base_path = Path(__file__).parent.parent
    queries = load_queries(base_path / "data" / "queries.jsonl")
    assert len(queries) > 0, "Queries should contain query entries"
    # Check a sample query has expected fields
    query = queries[0]
    assert hasattr(query, "id"), "Query should have an id"
    assert hasattr(query, "text"), "Query should have text"
    print(f"  Loaded {len(queries)} queries")


def test_load_qrels_every_id_exists_in_corpus_and_queries():
    """load_qrels parses the TSV, and every qrel ID exists in the corpus and queries."""
    base_path = Path(__file__).parent.parent
    qrels_path = base_path / "data" / "FinanceBench_qrels.tsv"

    # Load corpus and queries to validate against
    corpus = load_corpus(base_path / "data" / "corpus.jsonl")
    queries = load_queries(base_path / "data" / "queries.jsonl")

    corpus_ids = {doc.id for doc in corpus}
    query_ids = {q.id for q in queries}

    with qrels_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip header line
    data_lines = lines[1:] if lines and lines[0].startswith("query_id") else lines

    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        query_id = parts[0]
        corpus_id = parts[1]
        # Every qrel query ID must exist in queries
        assert query_id in query_ids, (
            f"qrel query_id {query_id} not found in queries.jsonl"
        )
        # Every qrel corpus ID must exist in corpus
        assert corpus_id in corpus_ids, (
            f"qrel corpus_id {corpus_id} not found in corpus.jsonl"
        )

    print(
        f"  Verified all {sum(1 for line in data_lines if line.strip())} qrel entries"
    )


def test_generate_answer_with_fake_model_returns_only_cited_references():
    """generate_answer with a fake model returns only the citations referenced in the answer text."""
    from rag.generator import generate_answer

    # Create a minimal fake model that returns a controlled answer
    class FakeModel:
        def __call__(self, prompt: str) -> str:
            # Return answer that references [1] and [3]
            return "The answer is [1] and [3]."

    # Create dummy chunks
    from rag.retriever import RetrievedChunk

    chunks = [
        RetrievedChunk(
            id="1", title="Doc 1", text="Content 1", score=1.0, source="bm25"
        ),
        RetrievedChunk(
            id="2", title="Doc 2", text="Content 2", score=0.9, source="bm25"
        ),
        RetrievedChunk(
            id="3", title="Doc 3", text="Content 3", score=0.8, source="bm25"
        ),
    ]

    answer = generate_answer("test query", chunks, FakeModel())

    # Answer references [1] and [3], so citations should only include those
    assert len(answer.citations) == 2, (
        f"Expected 2 citations, got {len(answer.citations)}"
    )
    citations_ids = {c.reference for c in answer.citations}
    assert 1 in citations_ids, "Should cite reference [1]"
    assert 3 in citations_ids, "Should cite reference [3]"
    assert 2 not in citations_ids, "Should NOT cite reference [2] (not in answer)"
    print(f"  Answer: {answer.answer}")
    print(f"  Citations: {answer.citations}")


def test_search_returns_503_when_api_key_unset(monkeypatch):
    from fastapi.testclient import TestClient

    from rag.retriever import RetrievedChunk
    from server import main

    class FakeRetriever:
        def retrieve(self, query, alpha=0.5):
            return [
                RetrievedChunk(id="d1", title="T", text="x", score=1.0, source="hybrid")
            ]

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(main, "_initialize_retriever", lambda: FakeRetriever())
    with TestClient(main.app) as client:
        response = client.post("/api/search", json={"query": "test"})
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
