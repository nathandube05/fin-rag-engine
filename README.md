# Financial RAG Engine

One-line pitch, then the problem: Analysts lose hours searching 10-K filings for specific figures; this answers questions over filings with cited sources.

## Problem
Financial analysts spend countless hours manually searching through 10-K filings and other corporate documents to find specific figures, trends, or insights. The fin-rag-engine solves this by providing a RAG (Retrieval-Augmented Generation) system that can answer questions over financial filings with proper source citations.

## How it works
1. **Hybrid Retrieval**: Combines BM25 (keyword-based) and dense embedding similarity in ChromaDB
2. **Alpha-weighted fusion**: BM25 and dense scores are combined with a weighted alpha parameter
3. **Min-max normalization**: Scores are normalized within each retrieval method before fusion
4. **Claude generates grounded answer**: The selected context is passed to Anthropic's Claude model, which generates an answer with inline citations formatted as [n]
5. **Citation filtering**: Only references actually mentioned in the answer are included

<details>
<summary>Click to see retrieval architecture diagram</summary>

```
query → BM25 + Dense Embeddings → ChromaDB
         ↓                  ↓
   scores α         scores (1-α)  →  Combine → Rerank → Top-k
         ↓                  ↓
   Min-Max Norm   Min-Max Norm
         ↓                  ↓
   Hybrid Scores → Retrieved Chunks → Claude → Answer with citations
```

</details>

## Evaluation

| Method            | Recall@10 | MRR@10 |
|-------------------|-----------|--------|
| BM25-only         | 0.3667    | 0.2113 |
| Dense-only        | 0.9889    | 0.6841 |
| Hybrid            | 0.9556    | 0.5540 |

Dense-only significantly outperformed hybrid retrieval on this FinanceBench benchmark with 180 documents (Recall@10: 0.9889 vs 0.9556, MRR@10: 0.6841 vs 0.5540). This is expected since k=10 retrieves a large share of the 180-document corpus, and the hybrid approach's alpha-weighted fusion can sometimes dilute the strong dense signals. The hybrid approach does provide complementary coverage to BM25-only, which recall@10 of only 0.3667 confirms is limited on this corpus.

**Next steps:** Tune the hybrid weighting (alpha) per query type and add a reranker model to improve hybrid performance. Do not claim hybrid is best on this benchmark.

## Quickstart
1. Clone the repository
2. Create a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Copy the example env file: `cp .env.example .env`
5. Add your Anthropic API key to `.env`
6. Start the server: `uvicorn server.main:app --reload`
7. Open http://localhost:8000

## Running tests
`pytest`

## Running evaluation
`python -m rag.evaluate`

## Deploying to Hugging Face Spaces
This application is configured for Hugging Face Spaces Docker deployment:

```yaml
sdk: docker
app_port: 7860
```

Add your `ANTHROPIC_API_KEY` as a Space secret. The Dockerfile pre-downloads the embedding model at build time and the app runs on port 7860.

## Project structure
```
fin-rag-engine/
├── data/              # FinanceBench data (corpus, queries, qrels)
├── client/            # Single-page frontend (HTML/JS/Tailwind)
├── docker-compose.yml # (optional) local development
├── Dockerfile         # HF Spaces deployment
├── requirements.txt   # Pinned dependencies
├── server/            # FastAPI backend
│   └── main.py
├── rag/               # RAG logic (ingestion, retriever, generator)
│   ├── ingestion.py
│   ├── generator.py
│   └── retriever.py
├── .env.example      # Environment variable template
├── .gitignore        # git ignore rules
└── README.md
```

## Limitations and next steps
- Whole-document chunks rather than finer chunking (could improve precision)
- No reranker model (could improve citation relevance)
- Retrieval-only evaluation (human evaluation needed for end-to-end quality)
- Single alpha value (could be learned/optimized per query type)

## Data credit
FinanceBench (Patronus AI)