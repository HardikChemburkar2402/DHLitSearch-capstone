import os
import json
import time
from pyalex import Works, config

config.email = os.getenv("OPENALEX_EMAIL", "hardikchemburkar@arizona.edu")

SEARCH_QUERIES = [
    "digital health literacy",
    "eHealth literacy",
    "online health information seeking",
    "mHealth patient engagement",
    "health information technology literacy"
]

def reconstruct_abstract(inverted_index: dict) -> str:
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)

def fetch_papers(query: str, max_results: int = 500) -> list:
    print(f"\nFetching: '{query}'...")
    papers = []
    results = (
        Works()
        .search(query)
        .filter(has_abstract=True)
        .select(["id", "doi", "title",
                 "abstract_inverted_index",
                 "publication_year",
                 "primary_location",
                 "authorships",
                 "concepts"])
        .paginate(per_page=200, n_max=max_results)
    )
    for page in results:
        for work in page:
            abstract = reconstruct_abstract(
                work.get("abstract_inverted_index", {})
            )
            if not abstract:
                continue
            papers.append({
                "id":       work.get("id", ""),
                "doi":      work.get("doi", ""),
                "title":    work.get("title", ""),
                "abstract": abstract,
                "year":     work.get("publication_year", None),
                "journal":  (work.get("primary_location") or {})
                            .get("source", {}).get("display_name", ""),
                "authors":  [a["author"]["display_name"]
                             for a in work.get("authorships", [])],
                "concepts": [c["display_name"]
                             for c in work.get("concepts", [])
                             if c.get("score", 0) > 0.3]
            })
        print(f"  Collected {len(papers)} papers so far...")
        time.sleep(0.5)
    return papers

def save_raw(papers: list, filename: str = "data/raw/openalex_raw.json"):
    os.makedirs("data/raw", exist_ok=True)
    with open(filename, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"\nSaved {len(papers)} papers to {filename}")

if __name__ == "__main__":
    all_papers = []
    for query in SEARCH_QUERIES:
        results = fetch_papers(query, max_results=500)
        all_papers.extend(results)
        print(f"Total so far: {len(all_papers)} papers")
    save_raw(all_papers)
