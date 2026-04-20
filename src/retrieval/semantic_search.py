from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from src.embeddings.biobert_embed import embed_abstract
from src.embeddings.chromadb_store import search as chroma_search
from src.retrieval.ranking import re_rank_results


@dataclass
class SearchResult:
    paper_id: str
    title: str
    year: Optional[int]
    journal: str
    authors: str
    abstract: str
    distance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _year_in_range(year: Optional[int], year_range: Optional[Tuple[int, int]]) -> bool:
    if not year_range:
        return True
    # Chroma metadata sometimes stores missing years as 0 (see chromadb_store.py).
    # Treat unknown/missing years as unbounded rather than excluding them, otherwise
    # semantic search can return zero results even when retrieval is working.
    if year is None or year == 0:
        return True
    start, end = year_range
    return start <= year <= end


def semantic_search(
    query: str,
    *,
    n_candidates: int = 100,
    k: int = 20,
    year_range: Optional[Tuple[int, int]] = None,
    # Many OpenAlex abstracts are short; a high default here can yield "empty" search UIs.
    min_words: int = 0,
    rerank: bool = True,
) -> List[SearchResult]:
    """
    Proposal-aligned retrieval entrypoint:
    - embed query with BioBERT
    - retrieve candidates from Chroma
    - optionally rerank using the hybrid heuristic
    - apply basic filters (year range, minimum length)
    """
    query_embedding = embed_abstract(query)
    raw = chroma_search(query_embedding, n_results=n_candidates)

    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    ids = raw.get("ids", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    if rerank:
        documents, metadatas, ids = re_rank_results(query, documents, metadatas, ids)
        # distances no longer aligned after rerank; keep None in that case
        distances = [None for _ in ids]

    results: List[SearchResult] = []
    for doc, meta, doc_id, dist in zip(documents, metadatas, ids, distances):
        abstract = (doc or "").strip()
        if min_words and len(abstract.split()) < min_words:
            continue

        title = (meta or {}).get("title", "") if isinstance(meta, dict) else ""
        year_val = (meta or {}).get("publication_year", (meta or {}).get("year")) if isinstance(meta, dict) else None
        try:
            year_int = int(year_val) if year_val not in (None, "") else None
        except Exception:
            year_int = None
        if year_int == 0:
            year_int = None

        if not _year_in_range(year_int, year_range):
            continue

        journal = (meta or {}).get("journal", "") if isinstance(meta, dict) else ""
        authors = (meta or {}).get("authors", "") if isinstance(meta, dict) else ""

        results.append(
            SearchResult(
                paper_id=str(doc_id),
                title=str(title or ""),
                year=year_int,
                journal=str(journal or ""),
                authors=str(authors or ""),
                abstract=abstract,
                distance=dist if isinstance(dist, (int, float)) else None,
            )
        )

        if len(results) >= k:
            break

    return results

