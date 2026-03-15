import json
import os
import chromadb

INPUT_PATH = "data/processed/embeddings.json"
CHROMA_DIR = ".chroma"
COLLECTION  = "dhlitsearch_abstracts"

def load_and_store():
    print("📂 Loading embeddings...")
    with open(INPUT_PATH) as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data)} embedded papers")

    # initialise persistent ChromaDB
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}   # cosine similarity
    )

    print("📥 Storing into ChromaDB — this takes ~2-3 mins...")

    ids         = []
    embeddings  = []
    metadatas   = []
    documents   = []

    for doc_id, record in data.items():
        ids.append(str(doc_id))
        embeddings.append(record["embedding"])
        documents.append(record.get("abstract", ""))
        metadatas.append({
            "title":   str(record.get("title",   ""))[:500],
            "year":    int(record.get("year")  or 0),
            "journal": str(record.get("journal",""))[:200],
            "authors": ", ".join(record.get("authors", []))[:300],
        })

    # ChromaDB recommends batching in chunks of 500
    batch_size = 500
    total      = len(ids)

    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        collection.add(
            ids        = ids[i:batch_end],
            embeddings = embeddings[i:batch_end],
            documents  = documents[i:batch_end],
            metadatas  = metadatas[i:batch_end]
        )
        print(f"  ✅ Stored {batch_end}/{total} papers...")

    print(f"\n🎉 Done! ChromaDB collection '{COLLECTION}' ready")
    print(f"   Total documents in store : {collection.count()}")
    return client, collection


def search(query_embedding: list, n_results: int = 10):
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION)
    results    = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results


if __name__ == "__main__":
    load_and_store()