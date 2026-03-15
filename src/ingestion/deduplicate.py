import json
import os

def deduplicate(input_path: str = "data/raw/openalex_raw.json",
                output_path: str = "data/processed/papers_deduplicated.json"):

    os.makedirs("data/processed", exist_ok=True)

    with open(input_path) as f:
        papers = json.load(f)

    print(f"Total before deduplication: {len(papers)}")

    seen_dois   = set()
    seen_titles = set()
    unique      = []
    duplicates  = 0

    for paper in papers:
        doi   = (paper.get("doi") or "").strip().lower()
        title = (paper.get("title") or "").strip().lower()

        if doi and doi in seen_dois:
            duplicates += 1
            continue
        if not doi and title in seen_titles:
            duplicates += 1
            continue

        if doi:
            seen_dois.add(doi)
        if title:
            seen_titles.add(title)
        unique.append(paper)

    print(f"Duplicates removed:         {duplicates}")
    print(f"Unique papers remaining:    {len(unique)}")

    with open(output_path, "w") as f:
        json.dump(unique, f, indent=2)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    deduplicate()
