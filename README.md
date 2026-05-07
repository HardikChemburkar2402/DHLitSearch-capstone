# DHLitSearch 🔬

### Digital Health Literacy Analysis with Semantic Search & Question Answering

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/BioBERT-Embeddings-FF6B6B?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/ChromaDB-10GB%20Vector%20Store-4ECDC4?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/RAG-Vertex%20AI%20%7C%20Gemini%20%7C%20Ollama-845EC2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Complete-39D353?style=for-the-badge"/>
</p>

<p align="center">
  <b>MS Data Science Capstone — Spring 2026</b><br/>
  Mentored by <b>Dr. Xiao Hu</b> &nbsp;|&nbsp; Built by <b>Hardik Chemburkar · Adaria Blackwell · Ralph Dsouza</b><br/>
  University of Arizona
</p>

---

## 🧠 The Problem

Every researcher knows the feeling.

You sit down to start a literature review. You type a few keywords into PubMed or Google Scholar. You get back 4,000 results. You spend the next three weeks clicking through abstract after abstract — most of them irrelevant — trying to piece together what the field actually knows about your topic.

**Systematic reviews in digital health commonly require screening between 5,000 and 10,000 papers.** Done manually, this process takes weeks or months. It is repetitive, error-prone, and pulls researchers away from the work that actually matters — analysis, synthesis, and discovery.

Existing tools fail to solve this adequately. PubMed's native search relies on keyword matching, which lacks semantic understanding. And while tools like ChatGPT seem promising, they have a fundamental flaw: **they have a fixed knowledge cutoff, cannot access current literature, and cannot cite specific sources.** Ask ChatGPT about the latest research on mHealth interventions and you get a confident, well-written, partially fabricated answer with no way to verify it.

**DHLitSearch was built to solve all three failure modes simultaneously.**

---

## 💡 What DHLitSearch Does

DHLitSearch is a domain-specific intelligent literature assistant for **digital health literacy research**. It doesn't just search — it understands, summarizes, answers, and maps the entire research landscape for you.

| Capability                      | What it means for you                                                                                                                                                                                                           |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🔍 **Semantic Search**          | Find papers by meaning, not just keywords — a search for "barriers to health app adoption in elderly patients" returns papers about "older adult mobile health acceptance" because BioBERT understands they mean the same thing |
| 📄 **Consolidated Summary**     | Automatically generates a cited AI summary across the top 5 ranked abstracts so you can judge overall relevance in seconds — no need to open each paper individually                                                            |
| 💬 **Question Answering (RAG)** | Ask natural language questions like _"What do recent studies say about digital divide and healthcare access?"_ — get a cited, grounded answer synthesized from real retrieved papers with explicit DOI-level citations          |
| 🗺️ **Topic Modeling**           | Discover dominant research themes across your retrieved corpus using BERTopic — with human-readable labels, a keyword-relevance-over-years chart, and a recent searches sidebar                                                 |
| 📥 **CSV Export**               | Download your full ranked result set as a structured CSV for downstream literature-review workflows                                                                                                                             |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                   Streamlit Web Application                     │
│              Q&A  ·  Search papers  ·  Themes                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │      Query Processing        │
          │   BioBERT Query Embedding    │
          └──────────────┬──────────────┘
                         │
     ┌───────────────────▼──────────────────────┐
     │           ChromaDB Vector Store           │
     │   490,482 Digital Health Literacy Papers  │
     │   BioBERT Embeddings  ·  ~10 GB on disk   │
     └──────┬──────────────────────┬─────────────┘
            │                      │
   ┌────────▼────────┐    ┌────────▼──────────────────────┐
   │  Search papers   │    │   Q&A — RAG Pipeline           │
   │  + Re-Ranking    │    │   Vertex AI → AI Studio        │
   │  + BART Summary  │    │   → Ollama → BART fallback     │
   └────────┬─────────┘    └────────┬───────────────────────┘
            │                       │
   ┌────────▼───────────────────────▼───────────────┐
   │   Results · Citations · Themes · CSV Export     │
   └─────────────────────────────────────────────────┘
```

---

## 🔬 Why BioBERT — Not Just Any Model

Most semantic search tools use general-purpose language models. DHLitSearch uses **BioBERT** (`dmis-lab/biobert-base-cased-v1.2`) — pre-trained specifically on PubMed abstracts and PMC full-text articles.

This matters because BioBERT understands that _"older adults"_ and _"elderly patients"_ are semantically equivalent in a clinical context, that _"health literacy intervention"_ is related to _"eHealth self-efficacy"_, and that _"mHealth"_ means mobile health — not just "mobile" and "health" as separate words. A general model misses all of this. BioBERT doesn't.

**Validation result: Top-1 cosine similarity averaged 92%+ across test queries. Precision@10 estimated at 70–80%.**

---

## 🗂️ Why RAG — Not ChatGPT

```
ChatGPT approach:
  Question → Language Model Memory → Answer
  ❌ Fixed knowledge cutoff — no access to current papers
  ❌ Cannot cite specific sources
  ❌ May hallucinate confident-sounding falsehoods

DHLitSearch RAG approach:
  Question → Retrieve Real Papers → LLM reads them → Cited Answer
  ✅ Every answer grounded in real retrieved documents
  ✅ Every claim linked to a specific paper DOI
  ✅ Out-of-scope queries rejected before LLM is called (distance threshold: 0.55)
  ✅ Prompt-level refusal instruction prevents fabrication on irrelevant context
```

---

## 📦 Tech Stack

```
Data Collection      OpenAlex API          19 queries · 414,000 +  papers · no key required
Embeddings           BioBERT               dmis-lab/biobert-base-cased-v1.2 · 768 dimensions
Vector Store         ChromaDB              ~10 GB · cosine similarity · persistent local store
Summarization        BART-large-cnn        Abstractive · 50-word minimum filter · Metal acceleration
LLM / RAG            Vertex AI (preferred) → AI Studio → Ollama (Llama 3) → BART fallback
Topic Modeling       BERTopic              Human-readable labels · keyword-over-years chart
Re-Ranking           Custom scorer         keyword_density × 2.0 + (year − 2000) × 0.5
Frontend             Streamlit             Three tabs: Q&A · Search papers · Themes
Deduplication        DOI-based             Title fallback for papers without DOI
```

---

## 📁 Repository Structure

```
dhlitsearch-capstone/
│
├── data/
│   ├── raw/                    # OpenAlex downloads (gitignored)
│   └── processed/              # Cleaned, deduplicated abstracts (gitignored)
│
├── src/
│   ├── ingestion/
│   │   ├── openalex_fetch.py   # Fetches papers from OpenAlex — checkpoint-based resumption
│   │   ├── preprocess.py       # Text cleaning, DOI normalization, abstract filtering
│   │   ├── deduplicate.py      # DOI-based deduplication across 19 query sets
│   │   └── stats.py            # Corpus landscape statistics
│   │
│   ├── embeddings/
│   │   ├── biobert_embed.py    # BioBERT embedding generation (768-dim, 0 failures)
│   │   └── chromadb_store.py   # Persistent ChromaDB storage and retrieval
│   │
│   ├── retrieval/
│   │   ├── semantic_search.py  # Cosine similarity search via ChromaDB
│   │   └── ranking.py          # Re-ranker: keyword density × 2 + year boost
│   │
│   ├── summarization/
│   │   └── bart_summarize.py   # BART-large-cnn abstractive summarization
│   │
│   ├── qa/
│   │   ├── rag_pipeline.py     # RAG orchestration + distance threshold + refusal
│   │   └── llm_interface.py    # Multi-backend: Vertex AI → AI Studio → Ollama → BART
│   │
│   └── topic_modeling/
│       └── bertopic_model.py   # BERTopic with human-readable labels
│
├── app/
│   └── streamlit_app.py        # Three-tab Streamlit application
│
│
├── docs/
│   ├── proposal.pdf
│   ├── gantt_chart.xlsx
│   ├── ethics_chart.xlsx
│   └── evaluation_report.md    # Precision@K and ROUGE results
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- ~12 GB free disk space (BioBERT model ~400MB + ChromaDB corpus ~10GB)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/hardikchemburkar/dhlitsearch-capstone.git
cd dhlitsearch-capstone

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Set up your environment variables
cp .env.example .env
# Add your OpenAlex email and Gemini / Vertex AI credentials
```

### Running the Pipeline

```bash
# Step 1 — Collect papers from OpenAlex (checkpoint-based, safe to interrupt)
python3 src/ingestion/openalex_fetch.py

# Step 2 — Preprocess and clean
python3 src/ingestion/preprocess.py

# Step 3 — Deduplicate
python3 src/ingestion/deduplicate.py

# Step 4 — View corpus statistics
python3 src/ingestion/stats.py

# Step 5 — Generate BioBERT embeddings (runs on CPU, ~several hours for full corpus)
python3 src/embeddings/biobert_embed.py

# Step 6 — Store in ChromaDB
python3 src/embeddings/chromadb_store.py

# Step 7 — Launch the app
streamlit run app/streamlit_app.py
```

---

## 📊 Corpus at a Glance

| Metric                  | Value                                        |
| ----------------------- | -------------------------------------------- |
| Total papers embedded   | 490,482                                      |
| Search queries used     | 19 targeted digital health literacy queries  |
| Primary data source     | OpenAlex API                                 |
| Year range              | 1959 – 2026                                  |
| Embedding model         | BioBERT (`dmis-lab/biobert-base-cased-v1.2`) |
| Embedding dimensions    | 768 per abstract                             |
| Vector store size       | ~10 GB (ChromaDB)                            |
| Embedding failures      | 0                                            |
| Papers without DOI      | 25,957 (deduplicated by title)               |
| Top-1 cosine similarity | 92%+ across test queries                     |
| Precision@10            | ~70–80% (manual evaluation)                  |

### Top Journals in Corpus

| Journal                                                           | Papers |
| ----------------------------------------------------------------- | ------ |
| PLoS ONE                                                          | 10,878 |
| International Journal of Environmental Research and Public Health | 6,272  |
| Journal of Medical Internet Research                              | 5,468  |
| BMJ Open                                                          | 5,052  |
| BMC Public Health                                                 | 4,750  |
| Frontiers in Psychology                                           | 4,009  |
| IEEE Access                                                       | 3,695  |
| Scientific Reports                                                | 3,500  |

---

## 📈 Pipeline Status

```
OpenAlex API  (19 queries)
        ↓
openalex_fetch.py     →  Raw papers collected with checkpoint resumption    ✅
        ↓
preprocess.py         →  Text cleaning, normalization, abstract filtering    ✅
        ↓
deduplicate.py        →  DOI-based deduplication                             ✅
        ↓
biobert_embed.py      →  490,482 × 768-dimensional embeddings               ✅  0 failures
        ↓
chromadb_store.py     →  Searchable vector database (~10 GB)                 ✅
        ↓
Semantic Search       →  92%+ cosine similarity on test queries              ✅
        ↓
BART Summarization    →  Consolidated top-5 overview + extractive fallback   ✅
        ↓
RAG Pipeline          →  Vertex AI → AI Studio → Ollama → BART fallback     ✅
        ↓
BERTopic Modeling     →  Human-readable labels + keyword-over-years chart    ✅
        ↓
Streamlit App         →  Q&A · Search papers · Themes — fully integrated     ✅
```

---

## 🗺️ Project Timeline

| Phase                           | Period          | Status                       |
| ------------------------------- | --------------- | ---------------------------- |
| Proposal & Planning             | Feb 18 – Feb 24 | ✅ Complete                  |
| Data Collection & Preprocessing | Feb 25 – Mar 9  | ✅ Complete                  |
| BioBERT Embedding Pipeline      | Mar 10 – Mar 16 | ✅ Complete                  |
| Summarization & RAG Pipeline    | Mar 17 – Mar 30 | ✅ Complete                  |
| Frontend (Streamlit)            | Mar 31 – Apr 13 | ✅ Complete                  |
| Post-Demo Professor Feedback    | Apr 14 – Apr 20 | ✅ Complete (commit 5092ccc) |
| Evaluation & Final Paper        | Apr 21 – Apr 27 | ✅ Complete                  |
| iShowcase Presentation          | May 10 – May 21 | ⏳ Upcoming                  |

---

## 👥 Team

| Name                  | Role            | Responsibilities                                                                   |
| --------------------- | --------------- | ---------------------------------------------------------------------------------- |
| **Hardik Chemburkar** | Project Manager | NLP pipeline, BioBERT embeddings, ChromaDB, semantic search, GitHub, documentation |
| **Adaria Blackwell**  | Data Engineer   | OpenAlex ingestion, preprocessing, deduplication, corpus stats, project poster     |
| **Ralph Dsouza**      | ML Engineer     | RAG/QA pipeline, LLM interface, BART summarization, re-ranking, Streamlit frontend |

**Faculty Advisor:** Dr. Xiao Hu &nbsp;|&nbsp; **Instructor:** Dr. Nitika Sharma

---

## 📄 Research Context

Digital health literacy — the ability to seek, find, understand, and appraise health information from electronic sources — is one of the fastest-growing interdisciplinary research areas in public health. The literature spans medicine, information science, psychology, and computer science, making it notoriously difficult to search comprehensively using any single keyword strategy.

DHLitSearch was designed specifically to address this interdisciplinary challenge. By combining domain-specific NLP models with a curated corpus from OpenAlex, the system surfaces connections across disciplines that traditional keyword search would miss entirely. Every answer is grounded in real retrieved documents — not generated from model memory — making the system trustworthy for academic use.

---

## ⚠️ Disclaimer

DHLitSearch is a research tool for academic literature review assistance. All AI-generated summaries and question-answering outputs are clearly labeled as AI-generated. This tool is **not** intended for clinical decision-making. Always consult original papers and qualified professionals for medical guidance.

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

<p align="center">
  Built with ☕ and 490,482 abstracts &nbsp;·&nbsp; Spring 2026<br/>
  <i>"The goal is to turn data into information, and information into insight."</i>
</p>
