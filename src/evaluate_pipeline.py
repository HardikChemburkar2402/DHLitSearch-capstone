import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rouge_score import rouge_scorer
from src.embeddings.biobert_embed import embed_abstract
from src.embeddings.chromadb_store import search
from src.summarization.bart_summarize import BartSummarizer


@dataclass
class RetrievalEvalRow:
    query: str
    k: int
    precision: float


@dataclass
class SummarizationEval:
    rouge1_f: float
    rouge2_f: float
    rougeL_f: float
    n_docs: int

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
        rows: List[RetrievalEvalRow] = []
        for item in self.test_queries:
            query = item["query"]
            keywords = item["keywords"]
            query_emb = embed_abstract(query)
            results = search(query_emb, n_results=k)
            documents = results.get("documents", [[]])[0]
            
            relevant_count = sum(1 for doc in documents if any(kw in doc.lower() for kw in keywords))
            precision = relevant_count / k
            rows.append(RetrievalEvalRow(query=query, k=int(k), precision=float(precision)))
            print(f"Query: '{query}' -> Precision@{k}: {precision * 100:.1f}%")
            
        avg_precision = statistics.mean([r.precision for r in rows]) if rows else 0.0
        print(f"Average Precision@{k}: {avg_precision * 100:.1f}%")
        return rows, float(avg_precision)

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
            return None
            
        r1_scores, r2_scores, rL_scores = [], [], []
        for i, original_abstract in enumerate(valid_documents):
            summary = self.summarizer.summarize(original_abstract)
            scores = self.scorer.score(original_abstract, summary)
            r1_scores.append(scores['rouge1'].fmeasure)
            r2_scores.append(scores['rouge2'].fmeasure)
            rL_scores.append(scores['rougeL'].fmeasure)

        if r1_scores:
            out = SummarizationEval(
                rouge1_f=float(statistics.mean(r1_scores)),
                rouge2_f=float(statistics.mean(r2_scores)),
                rougeL_f=float(statistics.mean(rL_scores)),
                n_docs=int(len(valid_documents)),
            )
            print(f"ROUGE-1: {out.rouge1_f:.3f}")
            print(f"ROUGE-2: {out.rouge2_f:.3f}")
            print(f"ROUGE-L: {out.rougeL_f:.3f}")
            return out
        else:
            print("Could not calculate ROUGE scores.")
            return None


def _format_report(*, retrieval_rows: List[RetrievalEvalRow], avg_precision: float, summarization: Optional[SummarizationEval]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: List[str] = []
    lines.append("# DHLitSearch — Pipeline Evaluation Report")
    lines.append("")
    lines.append(f"_Generated: {ts}_")
    lines.append("")
    lines.append("## Retrieval (Precision@k)")
    lines.append("")
    if retrieval_rows:
        lines.append("| Query | k | Precision@k |")
        lines.append("|---|---:|---:|")
        for r in retrieval_rows:
            lines.append(f"| {r.query} | {r.k} | {r.precision:.3f} |")
        lines.append("")
        lines.append(f"**Average Precision@{retrieval_rows[0].k}:** {avg_precision:.3f}")
    else:
        lines.append("No retrieval rows produced.")
    lines.append("")
    lines.append("## Summarization (ROUGE F1)")
    lines.append("")
    if summarization is None:
        lines.append("Summarization evaluation did not produce scores.")
    else:
        lines.append(f"- **Docs evaluated**: {summarization.n_docs}")
        lines.append(f"- **ROUGE-1 (F1)**: {summarization.rouge1_f:.3f}")
        lines.append(f"- **ROUGE-2 (F1)**: {summarization.rouge2_f:.3f}")
        lines.append(f"- **ROUGE-L (F1)**: {summarization.rougeL_f:.3f}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Retrieval relevance is estimated using simple keyword matching against returned abstracts (proxy metric).")
    lines.append("- ROUGE compares the generated summary to the original abstract (extractive faithfulness proxy).")
    lines.append("")
    return "\n".join(lines)


def generate_report(out_path: str = "docs/evaluation_report.md", *, k: int = 10) -> str:
    evaluator = PipelineEvaluator()
    retrieval_rows, avg_precision = evaluator.evaluate_retrieval(k=k)
    summarization = evaluator.evaluate_summarization()

    report_md = _format_report(
        retrieval_rows=retrieval_rows,
        avg_precision=avg_precision,
        summarization=summarization,
    )

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_md, encoding="utf-8")
    return str(path.resolve())

if __name__ == "__main__":
    report_path = generate_report(out_path="docs/evaluation_report.md", k=10)
    print(f"\n✅ Report written to: {report_path}\n")