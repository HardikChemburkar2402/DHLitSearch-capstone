import statistics
from rouge_score import rouge_scorer
from src.embeddings.biobert_embed import embed_abstract
from src.embeddings.chromadb_store import search
from src.summarization.bart_summarize import BartSummarizer

class PipelineEvaluator:
    def __init__(self):
        self.summarizer = BartSummarizer()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        self.test_queries = [
            {"query": "mHealth interventions for diabetes management", "keywords": ["mhealth", "mobile", "diabetes", "app", "smartphone"]},
            {"query": "eHealth literacy in older adults", "keywords": ["ehealth", "older", "elderly", "geriatric", "aging", "age"]},
            {"query": "telemedicine adoption barriers", "keywords": ["telemedicine", "telehealth", "barrier", "challenge", "adoption"]}
        ]

    def evaluate_retrieval(self, k=10):
        precisions = []
        for item in self.test_queries:
            query = item["query"]
            keywords = item["keywords"]
            query_emb = embed_abstract(query)
            results = search(query_emb, n_results=k)
            documents = results.get("documents", [[]])[0]
            
            relevant_count = sum(1 for doc in documents if any(kw in doc.lower() for kw in keywords))
            precision = relevant_count / k
            precisions.append(precision)
            print(f"Query: '{query}' -> Precision@{k}: {precision * 100:.1f}%")
            
        avg_precision = statistics.mean(precisions)
        print(f"Average Precision@{k}: {avg_precision * 100:.1f}%")
        return avg_precision

    def evaluate_summarization(self):
        query_emb = embed_abstract("impact of digital health literacy on patient outcomes and interventions")
        
        results = search(query_emb, n_results=500)
        raw_documents = results.get("documents", [[]])[0]
        
        valid_documents = []
        for doc in raw_documents:
            if len(doc.split()) >= 50:
                valid_documents.append(doc)
            
            if len(valid_documents) == 3:
                break
                
        if len(valid_documents) == 0:
            print("Error: Could not find any valid abstracts over 50 words for this query.")
            return 
            
        r1_scores, r2_scores, rL_scores = [], [], []
        for i, original_abstract in enumerate(valid_documents):
            summary = self.summarizer.summarize(original_abstract)
            scores = self.scorer.score(original_abstract, summary)
            r1_scores.append(scores['rouge1'].fmeasure)
            r2_scores.append(scores['rouge2'].fmeasure)
            rL_scores.append(scores['rougeL'].fmeasure)

        if r1_scores:
            print(f"ROUGE-1: {statistics.mean(r1_scores):.3f}")
            print(f"ROUGE-2: {statistics.mean(r2_scores):.3f}")
            print(f"ROUGE-L: {statistics.mean(rL_scores):.3f}")
        else:
            print("Could not calculate ROUGE scores.")

if __name__ == "__main__":
    evaluator = PipelineEvaluator()
    evaluator.evaluate_retrieval(k=10)
    evaluator.evaluate_summarization()