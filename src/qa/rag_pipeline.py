import time
from dotenv import load_dotenv

from src.retrieval.ranking import re_rank_results
from src.embeddings.biobert_embed import embed_abstract
from src.qa.llm_interface import GeminiLLM, OllamaLLM, LLMProviderError

load_dotenv()

# Cosine-distance cutoff for treating a query as out-of-corpus.
# Chroma is configured with hnsw:space=cosine, so distance ∈ [0, 2] where
# smaller = more similar. For BioBERT over the DHLit corpus, in-corpus queries
# typically land around 0.1–0.3; off-topic queries (quantum computing, finance,
# etc.) sit at ~0.45+. 0.55 is a conservative floor — raise to ~0.6 if real
# DHLit questions start getting rejected, lower to ~0.5 if junk still slips in.
OFF_CORPUS_DISTANCE = 0.55


class RAGPipeline:
    def __init__(self):
        # wrap provider init in try/except so the app still loads
        # even if a provider isn't installed or configured
        try:
            self.gemini = GeminiLLM(model="gemini-2.5-flash")
        except Exception:
            self.gemini = None
        try:
            self.ollama = OllamaLLM(model="llama3")
        except Exception:
            self.ollama = None
        self.local_summarizer = None

    def ask_question(self, query: str) -> str:
        from src.embeddings.chromadb_store import search
        query_embedding = embed_abstract(query)
        results = search(query_embedding, n_results=100)

        raw_documents = results.get("documents", [[]])[0]
        raw_metadatas = results.get("metadatas", [[]])[0]
        raw_ids = results.get("ids", [[]])[0]
        raw_distances = results.get("distances", [[]])[0]

        # Relevance floor: if even the nearest-neighbour paper is far from the
        # query embedding, the question is almost certainly outside the DHLit
        # corpus. Bail early instead of letting the LLM hallucinate a structured
        # answer over unrelated papers.
        if raw_distances:
            nearest = min(raw_distances)
            if nearest > OFF_CORPUS_DISTANCE:
                print(f"[Q&A] Off-corpus query (nearest distance={nearest:.3f}): {query!r}")
                return (
                    "This question appears to be outside the scope of the indexed "
                    "Digital Health Literacy corpus. No closely-related papers were "
                    "retrieved, so I'm not going to speculate.\n\n"
                    "Try rephrasing in terms of health literacy, eHealth, telemedicine, "
                    "patient engagement, health information seeking, or related topics."
                )

        ranked_docs, ranked_metas, ranked_ids = re_rank_results(query, raw_documents, raw_metadatas, raw_ids)
        
        valid_docs = []
        valid_metas = []
        valid_ids = []
        
        for doc, meta, doc_id in zip(ranked_docs, ranked_metas, ranked_ids):
            if len(doc.split()) >= 50:
                valid_docs.append(doc)
                valid_metas.append(meta)
                valid_ids.append(doc_id)
            
            if len(valid_docs) == 5:
                break
        
        if not valid_docs:
            return "I couldn't find any relevant, full-length papers in the database for that question."
            
        context = ""
        for i in range(len(valid_docs)):
            meta = valid_metas[i]
            title = meta.get("title", "Unknown Title")
            paper_id = valid_ids[i]
            abstract = valid_docs[i]
            
            context += f"[Paper ID: {paper_id}]\nTitle: {title}\nAbstract: {abstract}\n\n"
            
        prompt = f"""
        You are an expert biomedical research assistant answering questions about Digital Health Literacy.
        Answer the user's question based strictly on the provided context from research papers.

        RELEVANCE CHECK (do this first): if none of the provided papers directly address the user's question,
        respond with EXACTLY this single line and NOTHING else:
        "The indexed corpus does not contain papers that directly address this question."
        Do NOT produce a Key findings section. Do NOT list the retrieved paper IDs.
        Do NOT speculate or introduce outside knowledge.

        Otherwise, follow the format below.

        CITATIONS: You MUST include inline citations referencing the exact [Paper ID: ...] provided in the context
        (e.g., [Paper ID: https://doi.org/10.1234...]). DO NOT use generic citations like [Source 1] or [Source 2].

        Output format:
        - **Answer** (1–2 paragraphs, with inline [Paper ID: ...] citations where claims are made)
        - **Key findings** (3–5 bullets, each ending with a [Paper ID: ...] citation)

        Do NOT include any other sections (no "Evidence", no "Summary", no "Conclusion").

        Context:
        {context}

        Question: {query}
        """
        
        # retry on rate limits
        max_retries = 3
        if self.gemini is not None:
            for attempt in range(max_retries):
                try:
                    return self.gemini.generate(prompt).text
                except Exception:
                    if attempt < max_retries - 1:
                        print(f"⚠️ API High Demand (Attempt {attempt+1}/{max_retries}). Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        print("\n⚠️ Gemini unavailable. Falling back to local Ollama...\n")
                        break
                    
        # try local ollama if gemini is down
        if self.ollama is not None:
            try:
                return self.ollama.generate(prompt).text
            except LLMProviderError:
                print("\n⚠️ Ollama unavailable. Falling back to extractive evidence...\n")

        # last resort: extractive summarization
        if self.local_summarizer is None:
            from src.summarization.bart_summarize import BartSummarizer
            self.local_summarizer = BartSummarizer()

        evidence = "*(Fallback mode: showing evidence summaries — LLM generation unavailable)*\n\n"
        for i in range(min(3, len(valid_docs))):
            local_summary = self.local_summarizer.summarize(valid_docs[i])
            evidence += f"**Evidence {i+1} [Paper ID: {valid_ids[i]}]:**\n{local_summary}\n\n"
        return evidence
        
if __name__ == "__main__":
    bot = RAGPipeline()
    answer = bot.ask_question("How does eHealth literacy affect patient outcomes in older adults?")
    print(answer)
