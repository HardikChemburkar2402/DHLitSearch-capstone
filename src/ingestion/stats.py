import json
from collections import Counter

INPUT_PATH = "data/processed/papers_deduplicated.json"

def show_stats():
    with open(INPUT_PATH) as f:
        papers = json.load(f)

    print("\n" + "="*50)
    print("   📊 DHLitSearch — Corpus Statistics")
    print("="*50)

    # Total
    print(f"\n📚 Total unique papers     : {len(papers)}")

    # Year range
    years = [p["year"] for p in papers if p.get("year")]
    if years:
        print(f"📅 Year range              : {min(years)} – {max(years)}")

    # Papers per decade
    print(f"\n📈 Papers by decade:")
    decade_counts = Counter()
    for y in years:
        decade = (y // 10) * 10
        decade_counts[decade] += 1
    for decade in sorted(decade_counts):
        bar = "█" * (decade_counts[decade] // 10)
        print(f"   {decade}s : {bar} {decade_counts[decade]}")

    # Top journals
    journals = [p["journal"] for p in papers if p.get("journal")]
    print(f"\n🏛️  Top 10 journals:")
    for journal, count in Counter(journals).most_common(10):
        print(f"   {count:>4}  {journal}")

    # Top concepts/keywords
    all_concepts = []
    for p in papers:
        all_concepts.extend(p.get("concepts", []))
    print(f"\n🔑 Top 15 concepts/keywords:")
    for concept, count in Counter(all_concepts).most_common(15):
        print(f"   {count:>4}  {concept}")

    # Papers with no DOI
    no_doi = sum(1 for p in papers if not p.get("doi"))
    print(f"\n⚠️  Papers without DOI      : {no_doi}")
    print("\n" + "="*50)

if __name__ == "__main__":
    show_stats()