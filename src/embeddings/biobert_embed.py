import json
import os
import torch
from transformers import AutoTokenizer, AutoModel

INPUT_PATH  = "data/processed/papers_deduplicated.json"
OUTPUT_PATH = "data/processed/embeddings.json"

MODEL_NAME  = "dmis-lab/biobert-base-cased-v1.2"

print(f"🤖 Loading BioBERT model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def embed_abstract(text: str) -> list:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # mean pooling over all token embeddings
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embedding

def run():
    with open(INPUT_PATH) as f:
        papers = json.load(f)

    print(f"📚 Embedding {len(papers)} abstracts — this will take ~15-20 mins...")
    print("   Go grab a coffee ☕ — do not close the terminal\n")

    embeddings = {}
    failed     = 0

    for i, paper in enumerate(papers):
        doc_id   = paper.get("doi") or paper.get("id") or str(i)
        abstract = (paper.get("abstract") or "").strip()

        if not abstract:
            failed += 1
            continue

        try:
            embeddings[doc_id] = {
                "embedding": embed_abstract(abstract),
                "title":     paper.get("title", ""),
                "year":      paper.get("year"),
                "journal":   paper.get("journal", ""),
                "authors":   paper.get("authors", []),
                "concepts":  paper.get("concepts", []),
                "abstract":  abstract
            }
        except Exception as e:
            print(f"  ⚠️  Failed on paper {i}: {e}")
            failed += 1

        # progress every 100 papers
        if (i + 1) % 100 == 0:
            print(f"  ✅ Processed {i + 1}/{len(papers)} papers...")

    os.makedirs("data/processed", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(embeddings, f)

    print(f"\n🎉 Done!")
    print(f"   Embedded  : {len(embeddings)} papers")
    print(f"   Failed    : {failed} papers")
    print(f"   Saved to  : {OUTPUT_PATH}")

if __name__ == "__main__":
    run()
