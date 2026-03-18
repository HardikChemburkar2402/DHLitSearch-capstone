import os
import json
import time
from pyalex import Works, config

# ── Config ────────────────────────────────────────────────────────────────────
config.email = "hardikchemburkar@arizona.edu"  # replace with your real email

SEARCH_QUERIES = [
    "digital health literacy",
    "eHealth literacy",
    "online health information seeking",
    "mHealth patient engagement",
    "health information technology literacy"

    # NEW
    "telehealth literacy",
    "digital divide healthcare",
    "patient digital engagement",
    "health information seeking behavior",
    "eHealth interventions"
]

OUTPUT_PATH = "data/raw/openalex_raw.json"

# ── Abstract Reconstruction ───────────────────────────────────────────────────
# OpenAlex stores abstracts in inverted index format — this converts it back
def reconstruct_abstract(inverted_index: dict) -> str:
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)

# ── Fetch Papers ──────────────────────────────────────────────────────────────
def fetch_papers(query: str, max_results: int = 500) -> list:
    print(f"\n🔍 Fetching: '{query}'...")
    papers = []

    results = (
        Works()
        .search(query)
        .filter(has_abstract=True)
        .select([
            "id",
            "doi",
            "title",
            "abstract_inverted_index",
            "publication_year",
            "primary_location",
            "authorships",
            "concepts"
        ])
        .paginate(per_page=200, n_max=max_results)
    )

    for page in results:
        for work in page:
            # safely reconstruct abstract
            abstract = reconstruct_abstract(
                work.get("abstract_inverted_index") or {}
            )
            if not abstract:
                continue

            # safely get journal — primary_location can be None
            try:
                location = work.get("primary_location") or {}
                source   = location.get("source") or {}
                journal  = source.get("display_name", "")
            except Exception:
                journal = ""

            # safely get authors — authorships entries can be None
            try:
                authors = [
                    a["author"]["display_name"]
                    for a in (work.get("authorships") or [])
                    if a and a.get("author") and a["author"].get("display_name")
                ]
            except Exception:
                authors = []

            # safely get concepts
            try:
                concepts = [
                    c["display_name"]
                    for c in (work.get("concepts") or [])
                    if c and c.get("score", 0) > 0.3 and c.get("display_name")
                ]
            except Exception:
                concepts = []

            papers.append({
                "id":       work.get("id") or "",
                "doi":      work.get("doi") or "",
                "title":    work.get("title") or "",
                "abstract": abstract,
                "year":     work.get("publication_year"),
                "journal":  journal,
                "authors":  authors,
                "concepts": concepts
            })

        print(f"  ✅ Collected {len(papers)} papers so far...")
        time.sleep(0.5)

    return papers

# ── Save to File ──────────────────────────────────────────────────────────────
def save_raw(papers: list, path: str = OUTPUT_PATH):
    os.makedirs("data/raw", exist_ok=True)
    with open(path, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"\n💾 Saved {len(papers)} papers to {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_papers = []

    for query in SEARCH_QUERIES:
        results = fetch_papers(query, max_results=500)
        all_papers.extend(results)
        print(f"📊 Running total: {len(all_papers)} papers")
        time.sleep(1)

    save_raw(all_papers)
    print(f"\n🎉 Done! {len(all_papers)} total papers saved to {OUTPUT_PATH}")