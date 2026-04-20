import json
import os
from typing import Any, Tuple

INPUT_PATH  = "data/processed/papers_deduplicated.json"
OUTPUT_PATH = "data/processed/embeddings.json"

MODEL_NAME  = "dmis-lab/biobert-base-cased-v1.2"

_tokenizer = None
_model = None


def _get_biobert():
    """
    Lazy-load BioBERT to avoid downloads/import-time crashes and speed up UI startup.
    """
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print(f"🤖 Loading BioBERT model: {MODEL_NAME}")
        # Import heavy deps lazily to avoid hard crashes during Streamlit startup
        # on some macOS + Python builds.
        from transformers import AutoTokenizer, AutoModel  # type: ignore

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def _fallback_embedding(text: str, dim: int = 768) -> list:
    """
    Deterministic lightweight fallback when BioBERT/torch can't be imported.
    This keeps the UI usable (retrieval quality will be degraded).
    """
    import hashlib
    import math

    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    out = []
    for i in range(dim):
        b = h[i % len(h)]
        # map byte -> [-1, 1] with slight non-linearity
        x = (b / 255.0) * 2.0 - 1.0
        out.append(math.tanh(x))
    return out

def embed_abstract(text: str) -> list:
    try:
        # torch import is intentionally inside the call path
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
        # mean pooling over all token embeddings
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embedding
    except Exception as e:
        # If the environment can't support torch/transformers, keep app alive.
        print(f"⚠️ BioBERT embedding unavailable, using fallback embedding. ({e})")
        return _fallback_embedding(text)

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
