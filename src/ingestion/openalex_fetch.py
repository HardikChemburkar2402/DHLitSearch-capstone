import os
import json
import time
import random
from pyalex import Works, config

# ── Config ────────────────────────────────────────────────────────────────────
config.email                = "hardikchemburkar@arizona.edu"
config.max_retries          = 5
config.retry_backoff_factor = 0.5

SEARCH_QUERIES = [
    "digital health literacy",
    "eHealth literacy",
    "online health information seeking",
    "mHealth patient engagement",
    "health information technology literacy",
    "electronic health literacy",
    "internet health information",
    "digital health education",
    "mobile health literacy",
    "health app usability",
    "telemedicine patient literacy",
    "consumer health informatics",
    "patient digital engagement",
    "health website credibility",
    "online patient education",
    "telehealth literacy",
    "digital divide healthcare",
    "health information seeking behavior",
    "eHealth interventions"
]

OUTPUT_PATH = "data/raw/openalex_raw.json"
CHECKPOINT  = "data/raw/checkpoint.json"

# ── Abstract Reconstruction ───────────────────────────────────────────────────
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
def fetch_papers(query: str, max_results: int = 50000) -> list:
    print(f"\n🔍 Fetching: '{query}'...")
    papers = []

    try:
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
                abstract = reconstruct_abstract(
                    work.get("abstract_inverted_index") or {}
                )
                if not abstract:
                    continue

                try:
                    location = work.get("primary_location") or {}
                    source   = location.get("source") or {}
                    journal  = source.get("display_name", "")
                except Exception:
                    journal = ""

                try:
                    authors = [
                        a["author"]["display_name"]
                        for a in (work.get("authorships") or [])
                        if a and a.get("author") and a["author"].get("display_name")
                    ]
                except Exception:
                    authors = []

                try:
                    concepts = [
                        c["display_name"]
                        for c in (work.get("concepts") or [])
                        if c and c.get("score", 0) > 0.3 and c.get("display_name")
                    ]
                except Exception:
                    concepts = []

                papers.append({
                    "id":       work.get("id")    or "",
                    "doi":      work.get("doi")   or "",
                    "title":    work.get("title") or "",
                    "abstract": abstract,
                    "year":     work.get("publication_year"),
                    "journal":  journal,
                    "authors":  authors,
                    "concepts": concepts
                })

            print(f"  ✅ Collected {len(papers)} papers so far...")
            time.sleep(1.5)     # fast enough for progress, slow enough to avoid 429

    except Exception as e:
        print(f"\n  ⚠️  Stopped early on '{query}': {e}")
        print(f"  💾 Returning {len(papers)} papers collected before crash...")

    return papers

# ── Save checkpoint ───────────────────────────────────────────────────────────
def save_checkpoint(papers: list, completed: set):
    os.makedirs("data/raw", exist_ok=True)
    with open(CHECKPOINT, "w") as f:
        json.dump({
            "papers":            papers,
            "completed_queries": list(completed)
        }, f, indent=2)
    print(f"💾 Checkpoint saved — {len(papers):,} papers, {len(completed)} queries done")

# ── Save final output ─────────────────────────────────────────────────────────
def save_raw(papers: list, path: str = OUTPUT_PATH):
    os.makedirs("data/raw", exist_ok=True)
    with open(path, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"\n💾 Saved {len(papers):,} papers to {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Load checkpoint if exists ─────────────────────────────────────────────
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            ckpt = json.load(f)
        all_papers = ckpt["papers"]
        completed  = set(ckpt["completed_queries"])
        print(f"♻️  Resuming from checkpoint")
        print(f"   Papers collected : {len(all_papers):,}")
        print(f"   Queries done     : {len(completed)}/{len(SEARCH_QUERIES)}")
        print(f"   Queries left     : {len(SEARCH_QUERIES) - len(completed)}")
    else:
        all_papers = []
        completed  = set()
        print(f"🚀 Starting fresh run")
        print(f"   Total queries    : {len(SEARCH_QUERIES)}")
        print(f"   Max per query    : 50,000")

    # ── Run each query ────────────────────────────────────────────────────────
    for i, query in enumerate(SEARCH_QUERIES, 1):

        # skip already completed queries
        if query in completed:
            print(f"⏭️  [{i}/{len(SEARCH_QUERIES)}] Skipping '{query}' — already done")
            continue

        print(f"\n{'='*60}")
        print(f"  Query {i} of {len(SEARCH_QUERIES)}")
        print(f"{'='*60}")

        results = fetch_papers(query, max_results=50000)
        all_papers.extend(results)
        completed.add(query)

        print(f"\n📊 Running total : {len(all_papers):,} papers")
        print(f"   Queries done  : {len(completed)}/{len(SEARCH_QUERIES)}")

        # save checkpoint immediately after every query
        save_checkpoint(all_papers, completed)

        # short pause between queries
        wait = random.uniform(3.0, 5.0)
        print(f"⏳ Waiting {wait:.1f}s before next query...")
        time.sleep(wait)

    # ── Final save ────────────────────────────────────────────────────────────
    save_raw(all_papers)

    print(f"\n{'='*60}")
    print(f"🎉 ALL DONE!")
    print(f"   Total papers : {len(all_papers):,}")
    print(f"   Saved to     : {OUTPUT_PATH}")
    print(f"{'='*60}")

    # clean up checkpoint
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)
        print("🗑️  Checkpoint file cleaned up")