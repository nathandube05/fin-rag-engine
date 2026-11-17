from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from rag.generator import GeneratedAnswer, SourceCitation, generate_answer
from rag.ingestion import CorpusDocument, load_corpus
from rag.retriever import HybridRetriever, RetrievedChunk

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_PATH, ".env"), override=True)
logger = logging.getLogger(__name__)

CORPUS_PATH = os.path.join(BASE_PATH, "data", "corpus.jsonl")
CLIENT_INDEX = os.path.join(BASE_PATH, "client", "index.html")
CHROMA_PATH = os.getenv("RAG_CHROMA_PATH", os.path.join(BASE_PATH, "chroma_db"))
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SEARCH_TOP_K = int(os.getenv("RAG_SEARCH_TOP_K", "10"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RAG_RATE_LIMIT_PER_MINUTE", "10"))

SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer only from the supplied "
    "filing excerpts. Cite every supporting statement inline with its numbered "
    "reference, e.g. [1]. If the excerpts do not contain the answer, say so "
    "plainly. Never invent figures. Use plain text only, no markdown formatting."
)


# ---------- Request / response models ----------


class SearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=10_000)]
    mode: Annotated[Literal["hybrid", "dense", "bm25"], Field(default="hybrid")] = (
        "hybrid"
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class SearchResponse(BaseModel):
    query: str
    answer: str
    citations: list[SourceCitation]
    retrieved_chunks: list[RetrievedChunk]
    latency_ms: float
    model: str


# ---------- Language model ----------


class ModelConfigurationError(RuntimeError):
    pass


class AnthropicChatModel:
    """Callable wrapper around the Anthropic Messages API.

    The API key is read on every call, so the server can start without one
    and returns a clear 503 until it is configured.
    """

    def __call__(self, prompt: str) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ModelConfigurationError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file "
                "(see .env.example) to enable answer generation."
            )

        client = anthropic.Anthropic(
            api_key=api_key, base_url="https://api.anthropic.com"
        )
        try:
            message = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIStatusError as exc:
            raise RuntimeError(
                f"Anthropic API returned HTTP {exc.status_code}: {exc.message}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise RuntimeError(f"Could not reach the Anthropic API: {exc}") from exc

        text = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", "") == "text"
        ).strip()
        if not text:
            raise RuntimeError("Anthropic API returned an empty answer")
        return text


# ---------- Rate limiting (in-memory, per IP) ----------

_request_log: dict[str, deque[float]] = defaultdict(deque)


def _check_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    window = _request_log[client_ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a minute and try again.",
        )
    window.append(now)


# ---------- App setup ----------


def _initialize_retriever() -> HybridRetriever:
    if SEARCH_TOP_K <= 0:
        raise ValueError("RAG_SEARCH_TOP_K must be a positive integer")
    corpus: list[CorpusDocument] = load_corpus(CORPUS_PATH)
    return HybridRetriever(
        corpus=corpus,
        model_name=EMBEDDING_MODEL,
        chroma_path=CHROMA_PATH,
        k=SEARCH_TOP_K,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.retriever = _initialize_retriever()
    app.state.language_model = AnthropicChatModel()
    yield


app = FastAPI(
    title="Financial RAG Search API",
    description="Hybrid retrieval and cited answers over financial filings.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------- Routes ----------


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(CLIENT_INDEX)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request) -> SearchResponse:
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    alpha_map = {"hybrid": 0.5, "dense": 1.0, "bm25": 0.0}
    alpha = alpha_map[payload.mode]

    retriever: HybridRetriever = request.app.state.retriever
    chunks = retriever.retrieve(payload.query, alpha=alpha)
    try:
        generated: GeneratedAnswer = generate_answer(
            payload.query, chunks, request.app.state.language_model
        )
    except ModelConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    latency_ms = time.monotonic() - getattr(
        request.state, "start_time", time.monotonic()
    )

    return SearchResponse(
        query=payload.query,
        answer=generated.answer,
        citations=generated.citations,
        retrieved_chunks=chunks,
        latency_ms=latency_ms,
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
    )
