import os
import time
from google import genai
from dotenv import load_dotenv

from src.retrieval.ranking import re_rank_results
from src.embeddings.biobert_embed import embed_abstract
from src.embeddings.chromadb_store import search
from src.summarization.bart_summarize import BartSummarizer

load_dotenv()

class RAGPipeline:
    def __init__(self):
        self.client = genai.Client()
        self.local_fallback = None

    def ask_question(self, query: str) -> str:
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

        Context:
        {context}

        Question: {query}
        """
        
        # --- ROBUST RETRY LOGIC FOR PRESENTATIONS ---
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                return response.text
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ API High Demand (Attempt {attempt+1}/{max_retries}). Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print("\n🚨 Cloud LLM is down! Engaging local Fallback Mechanism on Apple Silicon...\n")
                    break
                    
        # --- LOCAL OFFLINE FALLBACK ---
        if self.local_fallback is None:
            self.local_fallback = BartSummarizer()
            
        fallback_answer = "⚠️ *Gemini LLM is currently unavailable due to API rate limits. Generating a local abstractive summary of retrieved documents using Apple Silicon...*\n\n"
        
        # Summarize the top 2 retrieved abstracts locally so it doesn't crash the presentation
        for i in range(min(2, len(valid_docs))):
            local_summary = self.local_fallback.summarize(valid_docs[i])
            fallback_answer += f"**Source {i+1} [Paper ID: {valid_ids[i]}]:**\n{local_summary}\n\n"
            
        return fallback_answer
        
if __name__ == "__main__":
    bot = RAGPipeline()
    answer = bot.ask_question("How does eHealth literacy affect patient outcomes in older adults?")
    print(answer)
