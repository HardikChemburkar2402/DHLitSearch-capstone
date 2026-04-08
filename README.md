# DHLitSearch 🔬

### Digital Health Literacy Analysis with Semantic Search & Question Answering

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/BioBERT-Embeddings-FF6B6B?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-4ECDC4?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/RAG-Question%20Answering-845EC2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-In%20Development-F9C74F?style=for-the-badge"/>
</p>

<p align="center">
  <b>A capstone research project — MS Data Science, Spring 2026</b><br/>
  Mentored by <b>Dr. Xiao Hu</b> &nbsp;|&nbsp; Built by <b>Hardik Chemburkar · Adaria Blackwell · Ralph Dsouza</b>
</p>

---

## 🧠 The Problem

Every researcher knows the feeling.

You sit down to start a literature review. You type a few keywords into PubMed or Google Scholar. You get back 4,000 results. You spend the next three weeks clicking through abstract after abstract — most of them irrelevant — trying to piece together what the field actually knows about your topic.

**Systematic reviews in digital health commonly require screening between 5,000 and 10,000 papers.** Done manually, this process takes weeks or months. It's repetitive, error-prone, and pulls researchers away from the work that actually matters — analysis, synthesis, and discovery.

Existing tools make it worse, not better. PubMed's keyword search doesn't understand meaning — it matches words, not concepts. And while tools like ChatGPT seem promising, they have a critical flaw: **they have a fixed knowledge cutoff, cannot access current literature, and cannot cite specific sources.** Ask ChatGPT about the latest research on mHealth interventions and you'll get a confident, well-written, partially fabricated answer with no way to verify it.

**DHLitSearch was built to solve all of this.**

---

## 💡 What DHLitSearch Does

DHLitSearch is a domain-specific intelligent literature assistant for **digital health literacy research**. It doesn't just search — it understands, summarizes, answers, and maps the entire research landscape for you.

| Capability                       | What it means for you                                                                                                                                                                    |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🔍 **Semantic Search**           | Find papers by meaning, not just keywords. Ask for "barriers to health app adoption in elderly patients" — get papers that discuss that concept even if they never use those exact words |
| 📄 **Abstractive Summarization** | Every retrieved paper comes with a 3–5 sentence AI-generated summary covering the objective, method, and key finding — no need to open each paper manually                               |
| 💬 **Question Answering (RAG)**  | Ask natural language questions like _"What do recent studies say about digital divide and healthcare access?"_ — get a cited, grounded answer synthesized from real retrieved papers     |
| 🗺️ **Topic Modeling**            | Discover the dominant themes across your entire retrieved corpus — see the research landscape at a glance using LDA or BERTopic                                                          |
| 📊 **Literature Statistics**     | Instantly see how many papers exist on your topic, their year distribution, top journals, and most frequent concepts                                                                     |
| 📥 **Export**                    | Download your results as a structured CSV for use in your own literature review document                                                                                                 |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                   Streamlit Web Application                     │
│         Search Bar · Filters · Results · QA Chat · Export       │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │      Query Processing        │
          │   BioBERT Query Embedding    │
          └──────────────┬──────────────┘
                         │
     ┌───────────────────▼──────────────────────┐
     │           ChromaDB Vector Store           │
     │   2,700+ Digital Health Literacy Papers   │
     │     Indexed by BioBERT Embeddings          │
     └──────┬──────────────────────┬─────────────┘
            │                      │
   ┌────────▼────────┐    ┌────────▼──────────────┐
   │  Semantic Search │    │   RAG Pipeline         │
   │  + Ranking       │    │   Gemini / Llama 3     │
   │  + Summarization │    │   Cited QA Answers     │
   └────────┬─────────┘    └────────┬───────────────┘
            │                       │
   ┌────────▼───────────────────────▼───────────────┐
   │              Results + Answers                  │
   │   Ranked Papers · Summaries · Citations         │
   │   Topic Map · Stats Dashboard · CSV Export      │
   └─────────────────────────────────────────────────┘
```

---

## 🔬 Why BioBERT — Not Just Any Model

Most semantic search tools use general-purpose language models. DHLitSearch uses **BioBERT** — a version of BERT pre-trained specifically on PubMed abstracts and PMC full-text articles.

This matters enormously for digital health literacy research. When a general model sees the phrase _"health literacy intervention"_, it treats it like any other English sentence. BioBERT has read millions of biomedical abstracts — it understands that _"intervention"_ in this context means a structured program, that _"health literacy"_ is a measurable outcome, and that this phrase is semantically related to _"eHealth self-efficacy"_ and _"patient empowerment"_ — even without keyword overlap.

The result: **search results that actually make sense to a domain researcher.**

---

## 🗂️ Why RAG — Not ChatGPT

Retrieval-Augmented Generation (RAG) is fundamentally different from asking an AI like ChatGPT a question.

```
ChatGPT approach:
  Question → Language Model Memory → Answer
  ❌ No access to current papers
  ❌ Cannot cite specific sources
  ❌ May hallucinate confident-sounding falsehoods

DHLitSearch RAG approach:
  Question → Retrieve Real Papers → LLM reads them → Cited Answer
  ✅ Answers grounded in real retrieved documents
  ✅ Every claim linked to a specific paper
  ✅ Always uses current literature, not training memory
```

This is the difference between a research assistant that has read the actual literature and one that is improvising from memory.

---

## 📦 Tech Stack

```
Data Collection     OpenAlex API          → 250M+ open academic papers, no key required
Embeddings          BioBERT (HuggingFace) → dmis-lab/biobert-base-cased-v1.2
Vector Store        ChromaDB              → Local persistent vector database
Summarization       BART / BioBART        → Abstractive summarization of abstracts
RAG / QA            Gemini API or Llama 3 → Cited question answering via Ollama
Topic Modeling      BERTopic / LDA        → Corpus-level theme extraction
Frontend            Streamlit             → Python-native web application
Deduplication       DOI-based             → Prevents duplicate papers across sources
```

---

## 📁 Repository Structure

```
dhlitsearch-capstone/
│
├── data/
│   ├── raw/                    # OpenAlex downloads (not committed to Git)
│   └── processed/              # Cleaned, deduplicated, embedded papers
│
├── src/
│   ├── ingestion/
│   │   ├── openalex_fetch.py   # Fetches papers from OpenAlex API
│   │   ├── preprocess.py       # Cleans and normalizes raw data
│   │   ├── deduplicate.py      # DOI-based deduplication across sources
│   │   └── stats.py            # Corpus landscape statistics
│   │
│   ├── embeddings/
│   │   ├── biobert_embed.py    # Generates BioBERT embeddings for abstracts
│   │   └── chromadb_store.py   # Stores and retrieves from ChromaDB
│   │
│   ├── retrieval/
│   │   ├── semantic_search.py  # Cosine similarity search against corpus
│   │   └── ranking.py          # Relevance ranking of results
│   │
│   ├── summarization/
│   │   └── bart_summarize.py   # BART abstractive summarization
│   │
│   ├── qa/
│   │   ├── rag_pipeline.py     # RAG orchestration
│   │   └── llm_interface.py    # Gemini / Ollama LLM interface
│   │
│   └── topic_modeling/
│       └── bertopic_model.py   # BERTopic / LDA topic extraction
│
├── app/
│   └── streamlit_app.py        # Main Streamlit web application
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_experiments.ipynb
│   └── 03_rag_experiments.ipynb
│
├── docs/
│   ├── proposal.pdf
│   ├── gantt_chart.xlsx
│   └── ethics_chart.xlsx
│
├── tests/
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
- ~5GB free disk space (for BioBERT model + corpus)

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
# Open .env and add your email and Gemini API key
```

### Running the Pipeline

```bash
# Step 1 – Collect papers from OpenAlex
python3 src/ingestion/openalex_fetch.py

# Step 2 – Preprocess and clean the corpus
python3 src/ingestion/preprocess.py

# Step 3 – Deduplicate the corpus
python3 src/ingestion/deduplicate.py

# Step 4 – View corpus statistics
python3 src/ingestion/stats.py

# Step 5 – Generate BioBERT embeddings (~20 mins first run)
python3 src/embeddings/biobert_embed.py

# Step 6 – Store embeddings in ChromaDB
python3 src/embeddings/chromadb_store.py

# Step 7 – Launch the Streamlit app
streamlit run app/streamlit_app.py
```

---

## 📊 Corpus at a Glance

> Stats generated from the processed and deduplicated corpus — updated as of March 2026

| Metric                     | Value                                      |
| -------------------------- | ------------------------------------------ |
| Total raw papers collected | 490,482+                                   |
| Primary data source        | OpenAlex (19 queries)                      |
| Embedding model            | BioBERT (dmis-lab/biobert-base-cased-v1.2) |
| Embedding dimensions       | 768                                        |
| Vector store               | ChromaDB (~6GB, cosine similarity)         |
| Top journal                | PLoS ONE (10,878 papers)                   |
| Papers without DOI         | 25,957                                     |

---

## 🗺️ Project Timeline

| Phase                           | Period          | Status         |
| ------------------------------- | --------------- | -------------- |
| Proposal & Planning             | Feb 18 – Feb 24 | ✅ Complete    |
| Data Collection & Preprocessing | Feb 25 – Mar 9  | ✅ Complete    |
| BioBERT Embedding Pipeline      | Mar 10 – Mar 16 | ✅ Complete    |
| Summarization & RAG Pipeline    | Mar 17 – Mar 30 | 🔄 In Progress |
| Frontend (Streamlit)            | Mar 31 – Apr 13 | ⏳ Upcoming    |
| Evaluation                      | Apr 14 – Apr 20 | ⏳ Upcoming    |
| Final Paper & Documentation     | Apr 21 – Apr 27 | ⏳ Upcoming    |
| iShowcase Presentation          | May 10 – May 21 | ⏳ Upcoming    |

---

## Where The Project Stands Right Now

```
OpenAlex API
  ↓
openalex_fetch.py      → 4,798 raw papers collected
  ↓
preprocess.py          → 4,728 cleaned & validated papers
  ↓
deduplicate.py         → 4,289 unique papers
  ↓
biobert_embed.py       → 4,289 × 768-dimensional embeddings
  ↓
chromadb_store.py      → Searchable vector database ✅
  ↓
Semantic Search        → ~92%+ relevance on first query ✅


---

## 👥 Team

| Name                  | Role            | Responsibilities                                               |
| --------------------- | --------------- | -------------------------------------------------------------- |
| **Hardik Chemburkar** | Project Manager | NLP pipeline, BioBERT embeddings, ChromaDB, semantic search    |
| **Adaria Blackwell**  | Data Engineer   | OpenAlex ingestion, preprocessing, deduplication, corpus stats |
| **Ralph Dsouza**      | ML Engineer     | Summarization, RAG/QA pipeline, topic modeling, Streamlit UI   |

**Faculty Advisor:** Dr. Xiao Hu
**Instructor:** Dr. Nitika Sharma
**Institution:** MS Data Science Capstone, Spring 2026

---

## 📄 Research Context

Digital health literacy — the ability to find, understand, and use digital health information effectively — is one of the most rapidly growing research areas in public health. The literature spans medicine, information science, psychology, and computer science, making it notoriously difficult to search comprehensively using any single keyword strategy.

DHLitSearch was designed specifically to address this interdisciplinary challenge. By combining domain-specific NLP models with a curated corpus from OpenAlex, the system can surface connections across disciplines that traditional keyword search would miss entirely.

---

## ⚠️ Disclaimer

DHLitSearch is a research tool built for academic literature review assistance. All AI-generated summaries and question-answering outputs are clearly labeled as AI-generated. This tool is **not** intended for clinical decision-making. Always consult the original papers and qualified professionals for medical or clinical guidance.

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

<p align="center">
  Built with ☕ and a lot of PubMed abstracts &nbsp;·&nbsp; Spring 2026<br/>
  <i>"The goal is to turn data into information, and information into insight."</i>
</p>
```
