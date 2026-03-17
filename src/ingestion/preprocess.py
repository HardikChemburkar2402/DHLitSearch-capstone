"""
DHLitSearch Capstone
Module: preprocess.py
Owner: Adaria Blackwell
Purpose:
    Clean and normalize raw OpenAlex literature records before deduplication,
    embeddings, and corpus statistics.

Input:
    data/raw/openalex_raw.json

Outputs:
    data/processed/papers_preprocessed.json
    data/processed/papers_preprocessed.csv
"""

import csv
import html
import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Tuple


INPUT_PATH = "data/raw/openalex_raw.json"
OUTPUT_JSON_PATH = "data/processed/papers_preprocessed.json"
OUTPUT_CSV_PATH = "data/processed/papers_preprocessed.csv"

CURRENT_YEAR = 2026


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters into a consistent representation."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


def clean_text(text: Any) -> str:
    """
    Clean free-text fields such as title and abstract.
    - Convert HTML entities
    - Normalize unicode
    - Remove excessive whitespace
    - Remove stray control characters
    """
    if text is None:
        return ""

    text = str(text)
    text = html.unescape(text)
    text = normalize_unicode(text)

    # Remove control characters except whitespace
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)

    # Normalize common dash/quote artifacts
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_doi(doi: Any) -> str:
    """
    Normalize DOI values into a consistent lowercase format.
    Removes URL prefixes like https://doi.org/
    """
    if doi is None:
        return ""

    doi = clean_text(doi).lower()

    doi = doi.replace("https://doi.org/", "")
    doi = doi.replace("http://doi.org/", "")
    doi = doi.replace("doi.org/", "")
    doi = doi.replace("doi:", "").strip()

    return doi


def normalize_openalex_id(record_id: Any) -> str:
    """Standardize the OpenAlex work ID field."""
    if record_id is None:
        return ""
    return clean_text(record_id)


def normalize_year(year: Any) -> int | None:
    """Keep only plausible publication years."""
    try:
        year = int(year)
        if 1900 <= year <= CURRENT_YEAR:
            return year
    except (TypeError, ValueError):
        pass
    return None


def normalize_string_list(values: Any) -> List[str]:
    """
    Clean, deduplicate, and preserve order for list-like string fields
    such as authors and concepts.
    """
    if not isinstance(values, list):
        return []

    cleaned = []
    seen = set()

    for value in values:
        item = clean_text(value)
        if not item:
            continue

        key = item.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(item)

    return cleaned


def normalize_journal(journal: Any) -> str:
    """Clean and standardize journal names."""
    journal = clean_text(journal)
    return journal


def validate_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate a preprocessed record.
    Returns:
        (is_valid, reason_if_invalid)
    """
    title = record.get("title", "")
    abstract = record.get("abstract", "")

    if not title:
        return False, "missing_title"
    if not abstract:
        return False, "missing_abstract"
    if len(title) < 5:
        return False, "short_title"
    if len(abstract) < 40:
        return False, "short_abstract"

    return True, ""


def preprocess_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Apply field-level cleaning and normalization to one paper record."""
    cleaned = {
        "id": normalize_openalex_id(record.get("id")),
        "doi": normalize_doi(record.get("doi")),
        "title": clean_text(record.get("title")),
        "abstract": clean_text(record.get("abstract")),
        "year": normalize_year(record.get("year")),
        "journal": normalize_journal(record.get("journal")),
        "authors": normalize_string_list(record.get("authors")),
        "concepts": normalize_string_list(record.get("concepts")),
    }

    # Helpful derived fields for downstream NLP / filtering
    cleaned["title_length"] = len(cleaned["title"])
    cleaned["abstract_length"] = len(cleaned["abstract"])
    cleaned["author_count"] = len(cleaned["authors"])
    cleaned["concept_count"] = len(cleaned["concepts"])

    return cleaned


def load_raw_papers(path: str) -> List[Dict[str, Any]]:
    """Load raw paper records from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(records: List[Dict[str, Any]], path: str) -> None:
    """Save records to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def save_csv(records: List[Dict[str, Any]], path: str) -> None:
    """Save records to CSV for easier inspection and downstream use."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "id",
        "doi",
        "title",
        "abstract",
        "year",
        "journal",
        "authors",
        "concepts",
        "title_length",
        "abstract_length",
        "author_count",
        "concept_count",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            row = record.copy()
            row["authors"] = "; ".join(record.get("authors", []))
            row["concepts"] = "; ".join(record.get("concepts", []))
            writer.writerow(row)


def preprocess_corpus() -> List[Dict[str, Any]]:
    """
    Main preprocessing pipeline:
    1. Load raw OpenAlex records
    2. Normalize fields
    3. Filter invalid records
    4. Save JSON + CSV outputs
    """
    raw_papers = load_raw_papers(INPUT_PATH)
    print(f"📥 Loaded raw papers: {len(raw_papers)}")

    processed = []
    dropped_counts = {
        "missing_title": 0,
        "missing_abstract": 0,
        "short_title": 0,
        "short_abstract": 0,
    }

    for record in raw_papers:
        cleaned = preprocess_record(record)
        is_valid, reason = validate_record(cleaned)

        if not is_valid:
            dropped_counts[reason] += 1
            continue

        processed.append(cleaned)

    save_json(processed, OUTPUT_JSON_PATH)
    save_csv(processed, OUTPUT_CSV_PATH)

    print("\n✅ Preprocessing complete")
    print(f"📄 Valid records kept: {len(processed)}")
    print("🗑️ Dropped records:")
    for reason, count in dropped_counts.items():
        print(f"   - {reason}: {count}")

    print(f"\n💾 JSON saved to: {OUTPUT_JSON_PATH}")
    print(f"💾 CSV saved to:  {OUTPUT_CSV_PATH}")

    return processed


if __name__ == "__main__":
    preprocess_corpus()