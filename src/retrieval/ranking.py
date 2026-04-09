def re_rank_results(query: str, documents: list, metadatas: list, ids: list) -> tuple:
    keywords = [word.lower() for word in query.split() if len(word) > 3]
    
    ranked_results = []
    
    for doc, meta, doc_id in zip(documents, metadatas, ids):
        kw_score = sum(doc.lower().count(kw) for kw in keywords)
        
        year = meta.get("publication_year", meta.get("year", 2010))
        try:
            year = int(year)
        except (ValueError, TypeError):
            year = 2010
            
        year_boost = max(0, year - 2000) * 0.5 
        total_score = (kw_score * 2.0) + year_boost
        
        ranked_results.append({
            "score": total_score,
            "document": doc,
            "metadata": meta,
            "id": doc_id
        })
        
    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    
    sorted_docs = [item["document"] for item in ranked_results]
    sorted_metas = [item["metadata"] for item in ranked_results]
    sorted_ids = [item["id"] for item in ranked_results]
    
    return sorted_docs, sorted_metas, sorted_ids