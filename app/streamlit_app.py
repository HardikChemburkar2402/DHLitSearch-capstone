import streamlit as st
import os
import sys
import pandas as pd

# Ensure the parent directory is in the path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Keep startup imports light: importing torch/bertopic eagerly can crash Streamlit on some macOS setups.
    from src.qa.rag_pipeline import RAGPipeline
except ImportError:
    class RAGPipeline:
        def ask_question(self, query):
            import time
            time.sleep(1.5)
            return "⚠️ *Backend dependencies missing. This is a mockup answer allowing you to view and interact with the UI.*"

def _semantic_search(*args, **kwargs):
    from src.retrieval.semantic_search import semantic_search as _fn

    return _fn(*args, **kwargs)




def _fit_bertopic(*args, **kwargs):
    from src.topic_modeling.bertopic_modeling import fit_bertopic as _fn

    return _fn(*args, **kwargs)


DEMO_SEARCH_QUERIES = [
    ("Telehealth & access", "telehealth access barriers rural underserved populations digital health literacy"),
    ("eHealth literacy", "eHealth literacy measurement tools and validated questionnaires"),
    ("Health misinformation", "health misinformation social media health literacy interventions"),
    ("Patient portals", "patient portal usability adoption older adults health literacy"),
    ("RCT / interventions", "randomized controlled trial digital health literacy intervention effectiveness"),
    ("Systematic reviews", "systematic review digital health literacy eHealth"),
]

# Page Config
st.set_page_config(
    page_title="DHLitSearch",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css(os.path.join(os.path.dirname(__file__), 'style.css'))

# Initialize RAG Engine in Session State
@st.cache_resource
def get_rag_engine():
    return RAGPipeline()



bot = get_rag_engine()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("DHLitSearch")
    st.caption("Corpus retrieval · BioBERT · Chroma")
    st.markdown("##### Retrieval settings")
    
    max_docs = st.slider(
        "Maximum Documents to Retrieve",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="How many papers to return in the ranked list.",
    )
    year_range = st.slider(
        "Filter by Publication Year",
        min_value=2000,
        max_value=2026,
        value=(2010, 2026),
        help="Timeframe constraint for semantic search.",
    )
    
    st.markdown("---")
    st.markdown("""
        <div class="ethics-disclaimer">
            <strong>Research use only.</strong> Not for clinical decision-making or patient care. Verify citations in primary sources.
        </div>
    """, unsafe_allow_html=True)

# --- MAIN LAYOUT (TABS) ---
st.markdown(
    """
    <div class="app-hero">
        <span class="tagline">Digital health literacy · Semantic retrieval</span>
        <h1>DHLitSearch workspace</h1>
        <p>Ask evidence-grounded questions, run semantic search over your embedded corpus, and explore themes with BERTopic— in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Semantic Q&A", "Semantic search", "Theme explorer"])

with tab1:
    st.markdown(
        """
        <div class="ui-card">
            <strong>RAG Q&amp;A</strong> retrieves relevant abstracts, then answers with inline cues to the evidence chain.
            Phrase questions like you would for a colleague (e.g. barriers, interventions, measurement).
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("##### Conversation")
    # Display Chat History 
    # Wrap in a container to push input to bottom if necessary
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about digital health literacy, telehealth, interventions, measurement…"):
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Synthesizing biomedical research..."):
                try:
                    response = bot.ask_question(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"An error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

with tab2:
    st.markdown(
        """
        <div class="ui-card">
            <strong>Semantic search</strong> embeds your query with BioBERT, retrieves candidates from Chroma, and optionally re-ranks with keyword overlap and recency.
            Use the quick-start buttons to populate the search box, then press <strong>Enter</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("##### Semantic search")
    st.caption("Query is embedded with BioBERT; results come from Chroma, optionally re-ranked for keyword overlap and recency.")

    st.markdown("**Quick start** — fills the search box; submit the search when ready.")
    for row_start in range(0, len(DEMO_SEARCH_QUERIES), 4):
        dq_cols = st.columns(4)
        for j, col in enumerate(dq_cols):
            idx = row_start + j
            if idx >= len(DEMO_SEARCH_QUERIES):
                break
            label, full_q = DEMO_SEARCH_QUERIES[idx]
            with col:
                if st.button(label, key=f"demo_q_{idx}", use_container_width=True, type="secondary"):
                    st.session_state["semantic_search_query"] = full_q
                    st.rerun()

    def _run_semantic_search_from_state():
        q = str(st.session_state.get("semantic_search_query", "")).strip()
        if not q:
            return
        with st.spinner("Searching ChromaDB and ranking results..."):
            results = _semantic_search(
                q,
                n_candidates=max(100, max_docs),
                k=max_docs,
                year_range=year_range,
                min_words=int(st.session_state.get("min_words", 0)),
                rerank=bool(st.session_state.get("rerank", True)),
            )
            st.session_state["last_search_query"] = q
            st.session_state["last_search_rerank"] = bool(st.session_state.get("rerank", True))
            st.session_state["last_search_results"] = [r.to_dict() for r in results]

    with st.form("semantic_search_form", clear_on_submit=False):
        query = st.text_input(
            "Search query",
            placeholder="e.g., barriers to telemedicine in rural areas with low health literacy",
            key="semantic_search_query",
        )
        col_a, col_b = st.columns([1, 1])
        with col_a:
            rerank = st.checkbox("Hybrid re-rank (keywords + recency)", value=True, key="rerank")
        with col_b:
            min_words = st.number_input(
                "Minimum abstract length (words)",
                min_value=0,
                max_value=200,
                value=0,
                step=5,
                help="OpenAlex abstracts are often short. Start at 0 for best recall; increase if you want longer abstracts only.",
                key="min_words",
            )

        # Keep a submit control for Enter-to-submit, but hide it via CSS to avoid UI redundancy.
        submitted = st.form_submit_button(" ", use_container_width=True, type="primary")
        if submitted:
            _run_semantic_search_from_state()

    results_dicts = st.session_state.get("last_search_results", [])
    last_q = st.session_state.get("last_search_query", "")
    if results_dicts:
        df = pd.DataFrame(results_dicts)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Results shown", len(df))
        with m2:
            st.metric("Last query length", len(last_q.split()) if last_q else 0, help="Words in the last submitted query")
        with m3:
            lr = st.session_state.get("last_search_rerank", rerank)
            st.metric("Re-ranking", "On" if lr else "Off")
        st.caption(f"Last query: _{last_q}_")
        
        st.markdown("##### Overview (top 5 papers)")
        if "top5_overview" not in st.session_state or st.session_state.get("overview_query") != last_q:
            top5 = df.head(5)
            context = ""
            for i, row in top5.iterrows():
                authors = row.get("authors", "")
                year = row.get("year", "")
                
                auth_list = str(authors).split(",")
                if len(auth_list) > 1:
                    # e.g. "Emeka B. Okeke" -> "Okeke et al."
                    cite_name = f"{auth_list[0].strip().split()[-1]} et al."
                elif len(auth_list) == 1 and auth_list[0]:
                    cite_name = auth_list[0].strip().split()[-1]
                else:
                    cite_name = "Unknown"
                    
                paper_id = row.get("paper_id", "#")
                # Format as a markdown link
                citation_key = f"[{cite_name}, {year}]({paper_id})"
                context += f"Citation Key: {citation_key}\nTitle: {row.get('title')}\nAbstract: {row.get('abstract')}\n\n"
            
            prompt = f"Based on the following top 5 research papers retrieved for the query '{last_q}', provide a concise, factual 1-paragraph overview of their main findings. When referencing a paper, you MUST cite it inline using its exact Citation Key (which is a clickable markdown link) provided below. Do not use [Paper 1] or introduce any external information.\n\n{context}"
            
            with st.spinner("Synthesizing top 5 papers..."):
                response = "⚠️ Overview generation requires Gemini API or Ollama to be configured."
                try:
                    if bot.gemini is not None:
                        response = bot.gemini.generate(prompt).text
                    elif bot.ollama is not None:
                        response = bot.ollama.generate(prompt).text
                except Exception as e:
                    response = f"⚠️ Overview generation failed: {e}"
            
            st.session_state["top5_overview"] = response
            st.session_state["overview_query"] = last_q
            
        st.info(st.session_state["top5_overview"])

        st.markdown("##### Ranked list")
        st.dataframe(
            df[["paper_id", "title", "year", "journal", "authors"]].fillna(""),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="dhlitsearch_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("##### Abstracts")
        for i, row in df.head(10).iterrows():
            title = row.get("title") or "Untitled"
            paper_id = row.get("paper_id")
            year = row.get("year")
            with st.expander(f"{title} ({year}) — {paper_id}"):
                st.markdown(f"**Journal:** {row.get('journal','')}")
                st.markdown(f"**Authors:** {row.get('authors','')}")
                st.markdown("**Abstract:**")
                st.write(row.get("abstract", ""))
    elif last_q:
        st.warning("No rows passed your filters. Try **minimum abstract length = 0** or widen the year range in the sidebar.")
    else:
        st.info("Submit a query above to see ranked papers, export CSV, and open abstracts here.")

with tab3:
    st.markdown(
        """
        <div class="ui-card">
            <strong>Theme explorer</strong> fits BERTopic on abstracts from your most recent semantic search, then shows the top topics and keywords.
            Run a search first, then come back here to explore themes.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("##### Theme explorer (BERTopic)")
    st.caption("Topics are fit on the **abstracts from your last semantic search** (rerun search if you change the corpus slice).")

    results_dicts = st.session_state.get("last_search_results", [])
    if not results_dicts:
        st.info("Run **Semantic search** first and click **Run search**. Then return here to discover themes in that result set.")
    else:
        df = pd.DataFrame(results_dicts)
        docs = [str(x) for x in df.get("abstract", []).tolist() if isinstance(x, str) and x.strip()]

        col1, col2 = st.columns([1, 1])
        with col1:
            nr_topics = st.number_input("Number of topics (optional)", min_value=0, max_value=50, value=0, step=1, help="0 = let BERTopic decide.")
        with col2:
            min_topic_size = st.number_input("Min topic size", min_value=2, max_value=50, value=10, step=1)

        if st.button("Run topic modeling", type="primary", use_container_width=True):
            with st.spinner("Fitting BERTopic (this may take a minute)..."):
                out = _fit_bertopic(
                    docs,
                    nr_topics=(None if int(nr_topics) == 0 else int(nr_topics)),
                    min_topic_size=int(min_topic_size),
                )
                st.session_state["topic_model_out"] = out

        out = st.session_state.get("topic_model_out")
        if out:
            summaries = out.get("summaries", [])
            if not summaries:
                st.warning("No topics found (or all documents were treated as outliers). Try lowering min topic size.")
            else:
                st.markdown("##### Top topics")
                topic_rows = []
                for s in summaries[:12]:
                    words = ", ".join([w for w, _ in s.top_words])
                    topic_rows.append({"topic_id": s.topic_id, "count": s.count, "top_words": words})
                st.dataframe(pd.DataFrame(topic_rows), use_container_width=True, hide_index=True)
