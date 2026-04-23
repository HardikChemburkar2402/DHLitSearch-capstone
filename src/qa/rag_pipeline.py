import time
from dotenv import load_dotenv

from src.retrieval.ranking import re_rank_results
from src.embeddings.biobert_embed import embed_abstract
from src.qa.llm_interface import GeminiLLM, OllamaLLM, LLMProviderError

load_dotenv()

class RAGPipeline:
    def __init__(self):
        # Keep Streamlit startup resilient: if a provider isn't installed/configured,
        # the app should still load and fall back gracefully at question-time.
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
        
        CRITICAL INSTRUCTION: You MUST include inline citations referencing the exact [Paper ID: ...] provided in the context (e.g., [Paper ID: https://doi.org/10.1234...]). 
        DO NOT use generic citations like [Source 1] or [Source 2].

        Output format:
        - Answer (1–2 paragraphs)
        - Key points (bullets)
        - Evidence (bullets, each ends with a [Paper ID: ...] citation)

        Context:
        {context}

        Question: {query}
        """
        
        # retry api on rate limits
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
                    
        # fallback to local ollama
        if self.ollama is not None:
            try:
                return self.ollama.generate(prompt).text
            except LLMProviderError:
                print("\n⚠️ Ollama unavailable. Falling back to extractive evidence...\n")

        # fallback to local extractive summarizer
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
