from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel

from rag.retriever import RetrievedChunk


class LanguageModel(Protocol):
    def invoke(self, input: str) -> object: ...


class SourceCitation(BaseModel):
    reference: int
    title: str
    chunk_id: str


class GeneratedAnswer(BaseModel):
    answer: str
    citations: list[SourceCitation]

    def render(self) -> str:
        if not self.citations:
            return self.answer
        source_lines = [
            f"[{citation.reference}] {citation.title} (chunk ID: {citation.chunk_id})"
            for citation in self.citations
        ]
        return f"{self.answer}\n\nSources:\n" + "\n".join(source_lines)


def _invoke_model(model: LanguageModel | object, prompt: str) -> str:
    invoke = getattr(model, "invoke", None)
    if callable(invoke):
        response = invoke(prompt)
    elif callable(model):
        response = model(prompt)
    else:
        raise TypeError("model must be callable or provide an invoke(prompt) method")

    content = getattr(response, "content", response)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("language model returned an empty or non-text response")
    return content.strip()


def _extract_cited_references(answer_text: str, max_reference: int) -> list[int]:
    """Extract [n] markers from the answer and return only those within range."""
    if not answer_text:
        return []
    markers = re.findall(r"\[(\d+)\]", answer_text)
    cited_refs = set()
    for m in markers:
        num = int(m)
        if 1 <= num <= max_reference:
            cited_refs.add(num)
    return sorted(cited_refs)


def generate_answer(
    query: str,
    retrieved_chunks: Sequence[RetrievedChunk],
    model: LanguageModel | object,
) -> GeneratedAnswer:
    if not query.strip():
        raise ValueError("query must not be empty")

    if not retrieved_chunks:
        return GeneratedAnswer(
            answer="I cannot answer because no supporting documents were retrieved.",
            citations=[],
        )

    context = "\n\n".join(
        f"[{index}] Title: {chunk.title}\nChunk ID: {chunk.id}\nContent:\n{chunk.text}"
        for index, chunk in enumerate(retrieved_chunks, start=1)
    )
    prompt = (
        "Answer the question using only the supplied financial document context. "
        "Do not invent facts. Cite supporting statements inline with the matching "
        "numbered reference, for example [1]. If the context does not contain "
        "enough information, say so.\n\n"
        f"Question:\n{query.strip()}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer:"
    )
    answer = _invoke_model(model, prompt)

    # Only cite references that actually appear in the answer
    max_ref = len(retrieved_chunks)
    cited_refs = _extract_cited_references(answer, max_ref)

    citations = [
        SourceCitation(
            reference=idx,
            title=chunk.title,
            chunk_id=chunk.id,
        )
        for idx, chunk in enumerate(retrieved_chunks, start=1)
        if idx in cited_refs
    ]
    return GeneratedAnswer(
        answer=answer,
        citations=citations,
    )
