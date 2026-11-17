# Financial Research Assistant

Ask a question about a company's SEC filings and get a short answer, with every figure linked back to the passage it came from.

![Financial Research Assistant](image-1.png)

## How it works

Take the question: *"How did Boeing's effective tax rate change from FY2021 to FY2022?"*

**1. Find the evidence**

The app searches 180 excerpts taken from company annual and quarterly reports (10-Ks and 10-Qs) and picks the 10 most relevant. You choose how it searches:

| Option | What it does | Good at |
|---|---|---|
| **Meaning** (default) | Compares what your question is *about* with what each passage is about | Finding the right passage even when it uses different words, e.g. "profitability" → "operating margin" |
| **Keywords** | Looks for passages that share words with your question, giving rarer words more weight | Exact terms, names and figures |
| **Both** | Scores each passage both ways and averages the two | Balancing the two approaches |

For Boeing, this pulls up the income statement from Boeing's 2022 annual report.

**2. Write the answer**

Those 10 passages, and nothing else, are given to Claude with three rules: only use what's in the passages, label every fact with the passage it came from, and say so if the answer isn't there.

Claude reads the income statement, works out the tax rate for each year (about 14.8% in 2021 and 0.6% in 2022), and notes that it calculated these figures because the filing doesn't state them directly.

**3. Show the sources**

Each fact in the answer carries a small numbered tag. Click it to jump to the exact passage it came from. Passages that were retrieved but not used are greyed out, so you can see exactly what the answer relied on.

<details>
<summary>Technical details</summary>

- **Meaning search:** passages are converted to vector embeddings with `all-MiniLM-L6-v2` (sentence-transformers) and stored in ChromaDB.
- **Keyword search:** BM25 via `rank-bm25`.
- **Both:** each method's scores are min-max normalised to 0–1, then averaged 50/50.
- **Generation:** Claude via the Anthropic API, with citations parsed from `[n]` markers and filtered to the ones actually used.

</details>
```

## Does it work? The evaluation

I tested retrieval on the 45 FinanceBench questions that come with relevance labels, measuring whether the right passage appears in the top 10 results (Recall@10) and how high it ranks (MRR@10).

| Search method      | Recall@10 | MRR@10 |
|--------------------|-----------|--------|
| Keywords (BM25)    | 0.367     | 0.211  |
| Meaning (dense)    | **0.989** | **0.684** |
| Both (hybrid)      | 0.956     | 0.554  |

The common advice is that hybrid search beats either method on its own. On this data it didn't. FinanceBench questions are long and wordy ("assume you are an analyst using only the cash flow statement…"), which drowns keyword search in irrelevant terms, and blending it in drags the hybrid score down. So the app defaults to meaning-based search.

Two caveats: the corpus is small (180 passages), so top 10 covers a fair share of it, and this measures retrieval only, not the quality of the final answers.

## Running it locally

```bash
git clone https://github.com/nathandube05/fin-rag-engine.git
cd fin-rag-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your ANTHROPIC_API_KEY
uvicorn server.main:app --reload
```

Open http://localhost:8000. The first start takes a minute while the embedding model downloads and the vector index builds.

```bash
pytest                      # tests (no API key needed)
python -m rag.evaluate      # reproduce the evaluation table
```

## Project structure

```
server/main.py     API: search endpoint, Claude call, rate limiting, serves the page
rag/retriever.py   meaning, keyword and hybrid search
rag/generator.py   prompt building and citation filtering
rag/evaluate.py    Recall@10 and MRR@10 on FinanceBench
rag/ingestion.py   loads the corpus and questions
client/index.html  the single-page interface
tests/             unit tests
data/              FinanceBench corpus, questions and relevance labels
```

## Limitations and what I'd do next

- **Arithmetic.** Claude does the maths itself, and it's occasionally loose (e.g. rounding a 14.2-point change to "roughly 15"). A calculator tool would fix this.
- **Whole-passage chunks.** Splitting long excerpts into smaller pieces would make retrieval and citations more precise.
- **No reranker.** A cross-encoder reranker on the top results is the obvious next step for ranking quality.
- **Answer quality isn't scored.** The evaluation covers retrieval; checking final answers against FinanceBench's reference answers would close the loop.
- **Small corpus.** 180 excerpts, not full filings.

## Built with

Python, FastAPI, ChromaDB, sentence-transformers, rank-bm25, the Anthropic API (Claude) and Tailwind CSS.

## Data and licence

Data comes from FinanceBench (Patronus AI, [arXiv:2311.11944](https://arxiv.org/abs/2311.11944)), via the [ICAIF '24 FinanceRAG Challenge](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge/data) on Kaggle. FinanceBench is released for non-commercial use.
