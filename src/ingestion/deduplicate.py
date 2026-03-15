import json
import os

INPUT_PATH  = "data/raw/openalex_raw.json"
OUTPUT_PATH = "data/processed/papers_deduplicated.json"

def deduplicate():
    # Load raw papers
    with open(INPUT_PATH) as f:
        papers = json.load(f)

    print(f"📥 Total papers before deduplication : {len(papers)}")

    seen_dois   = set()
    seen_titles = set()
    unique      = []
    duplicates  = 0

    for paper in papers:
        doi   = (paper.get("doi")   or "").strip().lower()
        title = (paper.get("title") or "").strip().lower()

        # skip if DOI already seen
        if doi and doi in seen_dois:
            duplicates += 1
            continue

        # skip if no DOI but title already seen
        if not doi and title in seen_titles:
            duplicates += 1
            continue

        # skip if no abstract
        if not (paper.get("abstract") or "").strip():
            duplicates += 1
            continue

        if doi:
            seen_dois.add(doi)
        if title:
            seen_titles.add(title)

        unique.append(paper)

    print(f"🗑️  Duplicates removed                : {duplicates}")
    print(f"✅ Unique papers remaining            : {len(unique)}")

    # Save
    os.makedirs("data/processed", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(unique, f, indent=2)

    print(f"💾 Saved to {OUTPUT_PATH}")
    return unique

if __name__ == "__main__":
    deduplicate()