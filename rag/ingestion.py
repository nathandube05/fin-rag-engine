from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field


class CorpusDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: str = Field(alias="_id", description="Unique document identifier")
    title: str = Field(default="", description="Document title or source filename")
    text: str = Field(description="Full document text content")


class Query(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: str = Field(alias="_id", description="Unique query identifier")
    title: str = Field(default="", description="Optional query title")
    text: str = Field(description="The query question or text")


ModelT = TypeVar("ModelT", bound=BaseModel)


def _load_jsonl(model_type: type[ModelT], path: Path | str) -> list[ModelT]:
    file_path = Path(path)
    records: list[ModelT] = []
    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(model_type.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {model_type.__name__} in {file_path} "
                    f"at line {line_number}: {exc}"
                ) from exc
    return records


def load_corpus(path: Path | str) -> list[CorpusDocument]:
    return _load_jsonl(CorpusDocument, path)


def load_queries(path: Path | str) -> list[Query]:
    return _load_jsonl(Query, path)


def ingest(
    corpus_path: Path | str, queries_path: Path | str
) -> tuple[list[CorpusDocument], list[Query]]:
    return load_corpus(corpus_path), load_queries(queries_path)


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    documents, queries = ingest(
        base_path / "data" / "corpus.jsonl",
        base_path / "data" / "queries.jsonl",
    )
    print(f"Loaded {len(documents)} corpus documents")
    print(f"Loaded {len(queries)} queries")
    if documents:
        print(
            f"Sample document: id={documents[0].id}, "
            f"title={documents[0].title!r}, text_len={len(documents[0].text)}"
        )
    if queries:
        print(f"Sample query: id={queries[0].id}, text={queries[0].text!r}")
