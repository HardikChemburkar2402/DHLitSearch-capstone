import json
import os

INPUT_PATH  = "data/processed/papers_deduplicated.json"
OUTPUT_PATH = "data/processed/embeddings.json"

MODEL_NAME  = "dmis-lab/biobert-base-cased-v1.2"

_tokenizer = None
_model = None


def _get_biobert():
    """Load BioBERT on first use so Streamlit startup stays fast and doesn't
    pull torch/transformers until we actually need to embed something."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print(f"Loading BioBERT model: {MODEL_NAME}")
        from transformers import AutoTokenizer, AutoModel  # type: ignore

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def _fallback_embedding(text: str, dim: int = 768) -> list:
    """Hash-based stand-in for BioBERT when torch/transformers can't load.

    Retrieval quality drops hard, but the UI stays usable instead of crashing
    the whole Streamlit process on import.
    """
    import hashlib
    import math

    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    out = []
    for i in range(dim):
        b = h[i % len(h)]
        x = (b / 255.0) * 2.0 - 1.0
        out.append(math.tanh(x))
    return out

def embed_abstract(text: str) -> list:
    try:
        import torch  # type: ignore

        tokenizer, model = _get_biobert()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # mean-pool token embeddings to get one vector per abstract
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embedding
    except Exception as e:
        print(f"BioBERT unavailable, using hash fallback. ({e})")
        return _fallback_embedding(text)

def run():
    with open(INPUT_PATH) as f:
        papers = json.load(f)

    print(f"Embedding {len(papers)} abstracts. This takes ~15-20 minutes on CPU.")
    print("Leave the terminal open until it finishes.\n")

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

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(papers)} papers...")

    os.makedirs("data/processed", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(embeddings, f)

    print("\nDone.")
    print(f"   Embedded  : {len(embeddings)} papers")
    print(f"   Failed    : {failed} papers")
    print(f"   Saved to  : {OUTPUT_PATH}")

if __name__ == "__main__":
    run()
