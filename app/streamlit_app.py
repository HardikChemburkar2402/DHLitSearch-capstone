import streamlit as st
import os
import sys
import time
import pandas as pd

# add parent dir so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # lazy import — torch/bertopic can be heavy and crash on some macOS setups
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

# page setup
st.set_page_config(
    page_title="DHLitSearch",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# inject custom stylesheet
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css(os.path.join(os.path.dirname(__file__), 'style.css'))

# single shared RAG engine (cached so it doesn't reload on every rerun)
@st.cache_resource
def get_rag_engine():
    return RAGPipeline()



bot = get_rag_engine()

# session defaults
if "messages" not in st.session_state:
    st.session_state.messages = []
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# --- sidebar controls ---
with st.sidebar:
    st.title("DHLitSearch")
    st.caption("Biomedical literature search")
    if getattr(bot, "gemini", None) is not None:
        try:
            st.caption(f"LLM: {bot.gemini.provider_label()}")
        except Exception:
            pass
    st.markdown("##### Retrieval settings")

    max_docs = st.slider(
        "Maximum papers to retrieve",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="How many papers to return in the ranked list.",
    )
    year_range = st.slider(
        "Publication year range",
        min_value=2000,
        max_value=2026,
        value=(2010, 2026),
        help="Only include papers published in this range.",
    )
    
    st.markdown("---")
    st.markdown("""
        <div class="ethics-disclaimer">
            <strong>Research use only.</strong> Not for clinical decision-making or patient care. Verify citations in primary sources.
        </div>
    """, unsafe_allow_html=True)

# --- main layout ---
st.markdown(
    """
    <div class="app-hero">
        <span class="tagline">Digital health literacy · Semantic retrieval</span>
        <h1>DHLitSearch workspace</h1>
        <p>Ask evidence-grounded questions, search the literature semantically, and explore themes — in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Q&A", "Search papers", "Themes"])

with tab1:
    st.markdown(
        """
        <div class="ui-card">
            <strong>RAG Q&amp;A</strong> retrieves relevant abstracts, then answers with citations linking back to the source papers.
            Phrase questions like you would for a colleague, e.g. <em>"What are the main barriers to telemedicine adoption in rural areas?"</em>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("##### Conversation")
    # render chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # handle new user input
    if prompt := st.chat_input("Ask about digital health literacy, telehealth, interventions, measurement…"):
        # save user message and show it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # get LLM answer
        with st.chat_message("assistant"):
            with st.spinner("Searching the literature and drafting an answer…"):
                try:
                    response = bot.ask_question(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    # Log the real traceback, but show a safe message in the UI.
                    print(f"[Q&A error] {type(e).__name__}: {e}")
                    user_msg = "Something went wrong while generating the answer. Please try again."
                    st.error(user_msg)
                    st.session_state.messages.append({"role": "assistant", "content": user_msg})

with tab2:
    st.markdown(
        """
        <div class="ui-card">
            <strong>Semantic search</strong> finds papers whose abstracts are meaning-similar to your query, then re-ranks by keyword overlap and recency.
            Use the quick-start buttons below, or type your own query and press <strong>Enter</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Quick start** — click a topic to fill the search box, then press **Enter** to search.")
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
            
            # save to search history (keep last 10)
            history_entry = {
                "id": f"search_{int(time.time())}",
                "timestamp": time.strftime("%I:%M %p"),
                "query": q,
                "results": st.session_state["last_search_results"]
            }
            st.session_state.search_history.insert(0, history_entry)
            st.session_state.search_history = st.session_state.search_history[:10]

    with st.form("semantic_search_form", clear_on_submit=False):
        query = st.text_input(
            "Search query",
            placeholder="e.g., barriers to telemedicine in rural areas with low health literacy",
            key="semantic_search_query",
        )
        col_a, col_b = st.columns([1, 1])
        with col_a:
            rerank = st.checkbox("Re-rank by keyword match and recency", value=True, key="rerank")
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

        # hidden submit button — lets Enter key trigger the form
        submitted = st.form_submit_button(" ", use_container_width=True, type="primary")
        if submitted:
            _run_semantic_search_from_state()

    results_dicts = st.session_state.get("last_search_results", [])
    last_q = st.session_state.get("last_search_query", "")
    if results_dicts:
        df = pd.DataFrame(results_dicts)
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Results shown", len(df))
        with m2:
            lr = st.session_state.get("last_search_rerank", rerank)
            st.metric("Re-ranking", "On" if lr else "Off")
        st.caption(f"Last query: _{last_q}_")
        
        st.markdown("##### Overview (top 5 papers)")
        top5 = df.head(5)
        top5_ids = "-".join(top5["paper_id"].astype(str).tolist())
        
        if "top5_overview" not in st.session_state or st.session_state.get("overview_paper_ids") != top5_ids:
            import re as _re_cite

            def _clean_surname(raw: object) -> str | None:
                """Return a usable surname or None if the authors field is garbage (PMID, parens, empty)."""
                if raw is None:
                    return None
                s = str(raw).strip()
                if not s or s.lower() in {"none", "nan", "unknown"}:
                    return None
                # Peel off parentheses/brackets/quotes so "(21736560)" doesn't become the surname.
                s = _re_cite.sub(r"[()\[\]{}\"']", "", s).strip()
                if not s:
                    return None
                token = s.split()[-1].strip(",.;:")
                if not token:
                    return None
                # Reject all-digit tokens (PMIDs, dates) and anything with < 2 alphabetic chars.
                if token.isdigit():
                    return None
                if sum(ch.isalpha() for ch in token) < 2:
                    return None
                return token

            _TITLE_STOPWORDS = {
                "a", "an", "the", "of", "on", "in", "to", "for", "and", "or", "with",
                "using", "via", "by", "from", "into", "at", "as", "is", "are", "be",
            }

            def _title_fallback(title: object) -> str | None:
                if not isinstance(title, str) or not title.strip():
                    return None
                for word in _re_cite.findall(r"[A-Za-z][A-Za-z\-]+", title):
                    if word.lower() not in _TITLE_STOPWORDS and len(word) >= 3:
                        return word
                return None

            context = ""
            for i, row in top5.iterrows():
                authors = row.get("authors", "")
                year = row.get("year", "")
                auth_list = [a for a in str(authors).split(",") if a.strip()]

                first_surname = _clean_surname(auth_list[0]) if auth_list else None
                if first_surname and len(auth_list) > 1:
                    cite_name = f"{first_surname} et al."
                elif first_surname:
                    cite_name = first_surname
                else:
                    # authors field is junk (PMID, empty, etc.) — use a title word
                    # so we never print a citation like "[21736560 et al., 2025]".
                    title_word = _title_fallback(row.get("title"))
                    cite_name = title_word or f"Paper {i + 1}"

                paper_id = row.get("paper_id", "#")
                citation_key = f"[{cite_name}, {year}]({paper_id})"
                context += (
                    f"Citation Key: {citation_key}\n"
                    f"Title: {row.get('title')}\n"
                    f"Abstract: {row.get('abstract')}\n\n"
                )

            prompt = (
                f"Based on the following top 5 research papers retrieved for the query '{last_q}', "
                "provide a concise, factual 1-paragraph overview of their main findings. "
                "When referencing a paper you MUST insert its Citation Key VERBATIM — copy it character-for-character "
                "exactly as shown below (including the surrounding square brackets and the parenthesised URL). "
                "Do NOT add parentheses around the surname, do NOT invent surnames, do NOT use placeholders "
                "like [Paper 1], and do NOT introduce any external information.\n\n"
                f"{context}"
            )
            
            with st.spinner("Synthesizing top 5 papers…"):
                response = "⚠️ The overview LLM is not configured. Skipping the auto-summary."
                for attempt in range(3):
                    try:
                        if bot.gemini is not None:
                            response = bot.gemini.generate(prompt).text
                            break
                        elif bot.ollama is not None:
                            response = bot.ollama.generate(prompt).text
                            break
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(2)
                        else:
                            err_str = str(e)
                            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                                response = "⏳ **Gemini API rate limit reached.** The free tier allows a limited number of requests per minute. Please wait ~30 seconds and refresh the page to try again."
                            else:
                                response = "⚠️ Overview generation is temporarily unavailable. Please try again in a moment."
            
            st.session_state["top5_overview"] = response
            st.session_state["overview_paper_ids"] = top5_ids
            
        st.info(st.session_state["top5_overview"])

        st.markdown("##### Ranked list")
        st.dataframe(
            df[["paper_id", "title", "year", "journal", "authors"]].fillna(""),
            use_container_width=True,
            hide_index=True,
            column_config={
                "paper_id": st.column_config.TextColumn("DOI / ID"),
                "title": st.column_config.TextColumn("Title"),
                "year": st.column_config.NumberColumn("Year", format="%d"),
                "journal": st.column_config.TextColumn("Journal"),
                "authors": st.column_config.TextColumn("Authors"),
            },
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
            with st.expander(f"{title} ({year})"):
                st.markdown(f"**Journal:** {row.get('journal','')}")
                st.markdown(f"**Authors:** {row.get('authors','')}")
                if paper_id:
                    st.markdown(f"**Link:** {paper_id}")
                st.markdown("**Abstract:**")
                st.write(row.get("abstract", ""))
    elif last_q:
        st.warning("No papers passed your filters. Try setting **Minimum abstract length** to 0, or widen the year range in the sidebar.")
    else:
        st.info("Type a query above (or click a quick-start button) and press **Enter** to see ranked papers, export the CSV, and open abstracts here.")

with tab3:
    st.markdown(
        """
        <div class="ui-card">
            <strong>Theme explorer</strong> groups a past search's papers into themes, and shows how each theme's top keywords trend over time.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Build the selectable search list from history + current last search
    history = list(st.session_state.get("search_history", []))
    last_results = st.session_state.get("last_search_results", [])
    last_query = st.session_state.get("last_search_query", "")

    # Ensure the current search (if any) appears even if not yet in history
    current_id = None
    if last_results and last_query:
        current_id = f"current::{last_query}"
        if not any(h.get("query") == last_query and h.get("results") == last_results for h in history):
            history = [{
                "id": current_id,
                "timestamp": "now",
                "query": last_query,
                "results": last_results,
            }] + history

    if not history:
        st.info("Run **Semantic search** first. Then return here to pick a search and explore its themes.")
    else:
        options = {}
        for h in history:
            label = f"{h.get('timestamp','?')} — {h.get('query','(unnamed search)')} ({len(h.get('results', []))} papers)"
            options[label] = h

        selected_label = st.selectbox(
            "Select a past search to analyze",
            options=list(options.keys()),
            index=0,
            help="Pick any search you've run this session.",
        )
        selected = options[selected_label]
        search_id = selected.get("id") or f"hist::{selected.get('query','')}::{selected.get('timestamp','')}"

        # Per-search caches
        if "topic_model_by_search" not in st.session_state:
            st.session_state["topic_model_by_search"] = {}
        if "topic_labels_by_search" not in st.session_state:
            st.session_state["topic_labels_by_search"] = {}

        # Build aligned (abstract, year) pairs so the topic assignments map back to years
        df_sel = pd.DataFrame(selected.get("results", []))
        aligned: list[tuple[str, object]] = []
        if not df_sel.empty:
            for _, row in df_sel.iterrows():
                abstract = row.get("abstract")
                if isinstance(abstract, str) and abstract.strip():
                    aligned.append((abstract, row.get("year")))

        docs = [a for a, _ in aligned]
        years = [y for _, y in aligned]

        out = st.session_state["topic_model_by_search"].get(search_id)
        run_clicked = st.button(
            "Discover themes in this search",
            type="primary",
            use_container_width=True,
            disabled=(len(docs) == 0),
        )

        if run_clicked:
            if len(docs) < 3:
                st.warning("Not enough abstracts in this search to find themes.")
            else:
                # Pick min_topic_size based on how many docs we have, and if BERTopic
                # buckets everything as outliers (happens on small or diverse sets),
                # drop the threshold and try again before giving up.
                first_guess = max(3, min(10, len(docs) // 5))
                candidates = sorted({first_guess, 5, 3}, reverse=True)

                out = None
                used_size = None
                with st.spinner("Finding themes in the selected search (this may take a minute)…"):
                    for mts in candidates:
                        attempt = _fit_bertopic(docs, nr_topics=None, min_topic_size=mts)
                        if attempt.get("summaries"):
                            out = attempt
                            used_size = mts
                            break
                        out = attempt  # keep last attempt so we still render the warning below

                if out is not None:
                    out["_years"] = years
                    out["_min_topic_size_used"] = used_size
                    st.session_state["topic_model_by_search"][search_id] = out
                    st.session_state["topic_labels_by_search"].pop(search_id, None)

        if not out:
            st.info("Click **Discover themes** above to find themes in this search.")
        else:
            summaries = out.get("summaries", [])
            if not summaries:
                st.warning(
                    "We couldn't find clear themes in this search — every paper ended up as an outlier, "
                    "even after auto-retrying with smaller cluster sizes. This usually means the result set is "
                    "too small or too topically diverse to cluster. Try running a broader or narrower "
                    "semantic search and reloading this tab."
                )
            else:
                # Generate human-readable topic labels (cached per search)
                labels = st.session_state["topic_labels_by_search"].get(search_id)
                if labels is None:
                    labels = {}
                    if getattr(bot, "gemini", None) is not None:
                        prompt = (
                            "Generate a short, 2-3 word human-readable label for each of the following research topics "
                            "based on their top keywords. Return EXACTLY one label per line, in the exact same order.\n\n"
                        )
                        for s in summaries[:12]:
                            prompt += f"Topic {s.topic_id}: {', '.join([w for w, _ in s.top_words[:10]])}\n"
                        with st.spinner("Generating theme labels…"):
                            for attempt in range(3):
                                try:
                                    response_text = bot.gemini.generate(prompt).text
                                    lines = [ln.strip() for ln in response_text.strip().split("\n") if ln.strip()]
                                    import re
                                    for idx, s in enumerate(summaries[:12]):
                                        if idx < len(lines):
                                            clean_label = re.sub(r'^(Topic\s*\d+:?|\d+\.|-|\*)\s*', '', lines[idx]).strip()
                                            labels[s.topic_id] = clean_label or " ".join(w.title() for w, _ in s.top_words[:2])
                                    break
                                except Exception:
                                    if attempt < 2:
                                        time.sleep(2)
                                    else:
                                        st.warning("Auto-labelling is unavailable — showing keyword-based theme labels instead.")
                                        for s in summaries[:12]:
                                            labels[s.topic_id] = " ".join(w.title() for w, _ in s.top_words[:2])
                    else:
                        # No LLM at all: keyword-based labels
                        for s in summaries[:12]:
                            labels[s.topic_id] = " ".join(w.title() for w, _ in s.top_words[:2])
                    st.session_state["topic_labels_by_search"][search_id] = labels

                # Top themes table (labels + counts + keywords)
                st.markdown("##### Top themes")
                topic_rows = []
                for s in summaries[:12]:
                    words = ", ".join([w for w, _ in s.top_words])
                    label = labels.get(s.topic_id, "Unknown Theme")
                    topic_rows.append({"Theme": label, "Papers": s.count, "Keywords": words})
                st.dataframe(pd.DataFrame(topic_rows), use_container_width=True, hide_index=True)

                _mts = out.get("_min_topic_size_used")
                if _mts is not None:
                    st.caption(
                        f"_Smallest theme contains **{_mts}** papers "
                        f"(auto-chosen based on the search size)._"
                    )

                # Theme picker (uses human-readable labels, maps back to topic_id internally)
                st.markdown("##### Explore a specific theme")
                topic_label_map = {}
                for s in summaries[:12]:
                    label = labels.get(s.topic_id, f"Theme {s.topic_id}")
                    display = f"{label} ({s.count} papers)"
                    # Avoid duplicate keys when two themes end up with the same label
                    if display in topic_label_map:
                        display = f"{display} — id {s.topic_id}"
                    topic_label_map[display] = s
                picked_display = st.selectbox("Select a theme", options=list(topic_label_map.keys()))
                picked_summary = topic_label_map[picked_display]

                # Always print the actual top keywords alongside the LLM-generated
                # label so the reviewer can verify the label matches the cluster.
                st.caption(
                    "**Top keywords for this theme:** "
                    + ", ".join(f"`{w}`" for w, _ in picked_summary.top_words[:5])
                )

                # Keyword-over-time chart for the picked theme
                topic_assignments = out.get("topics") or []
                year_list = out.get("_years") or []
                if len(topic_assignments) != len(year_list):
                    st.info("The trend chart can't line up papers with years for this run. Click **Discover themes** again to refresh.")
                else:
                    picked_id = picked_summary.topic_id
                    top_keywords = [w for w, _ in picked_summary.top_words[:5]]

                    # Gather (year, abstract) pairs for docs assigned to the picked topic
                    topic_docs = []
                    for doc_text, assigned_topic, y in zip(docs, topic_assignments, year_list):
                        if int(assigned_topic) == int(picked_id):
                            try:
                                y_int = int(y)
                            except (TypeError, ValueError):
                                continue
                            topic_docs.append((y_int, doc_text.lower()))

                    if not topic_docs or not top_keywords:
                        st.info("No year-tagged papers were assigned to this theme, so no trend chart can be drawn.")
                    else:
                        import re as _re
                        import altair as alt  # type: ignore

                        # Word-boundary regex so "health" doesn't match inside "healthcare".
                        # Bigrams like "health literacy" still work because re.escape
                        # preserves the space inside the keyword.
                        kw_patterns = {
                            kw: _re.compile(r"\b" + _re.escape(kw.lower()) + r"\b")
                            for kw in top_keywords
                        }

                        # Fill every year in the topic's range with zeros so the chart
                        # shows real gaps instead of drawing straight lines across them.
                        data_years = sorted({y for y, _ in topic_docs})
                        full_year_range = list(range(min(data_years), max(data_years) + 1))

                        rows = []
                        docs_per_year = {}
                        for y in full_year_range:
                            year_texts = [txt for yy, txt in topic_docs if yy == y]
                            docs_per_year[y] = len(year_texts)
                            for kw, pat in kw_patterns.items():
                                rows.append({
                                    "year": int(y),
                                    "keyword": kw,
                                    "mentions": sum(1 for txt in year_texts if pat.search(txt)),
                                })

                        trend_long = pd.DataFrame(rows)
                        st.markdown(f"**Papers mentioning top keywords over time** — _{picked_display}_")

                        y_max = max(1, int(trend_long["mentions"].max()))
                        chart = (
                            alt.Chart(trend_long)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X(
                                    "year:O",
                                    title="Year",
                                    axis=alt.Axis(labelAngle=0, format="d"),
                                ),
                                y=alt.Y(
                                    "mentions:Q",
                                    title="Papers mentioning keyword",
                                    scale=alt.Scale(domain=[0, y_max]),
                                    axis=alt.Axis(tickMinStep=1, format="d"),
                                ),
                                color=alt.Color("keyword:N", title="Keyword"),
                                tooltip=[
                                    alt.Tooltip("year:O", title="Year"),
                                    alt.Tooltip("keyword:N", title="Keyword"),
                                    alt.Tooltip("mentions:Q", title="Papers"),
                                ],
                            )
                            .properties(height=320)
                        )
                        st.altair_chart(chart, use_container_width=True)

                        st.caption(
                            "Counts are **papers** (not raw mentions) per year that contain the keyword "
                            "as a whole word. Missing years in the topic are shown as zero."
                        )

                        # Denominator: how many papers in the theme exist each year
                        denom_df = pd.DataFrame(
                            [{"year": int(y), "papers in theme": docs_per_year[y]} for y in full_year_range]
                        )

                        with st.expander("Show raw counts"):
                            wide = (
                                trend_long
                                .pivot(index="year", columns="keyword", values="mentions")
                                .reindex(full_year_range)
                                .fillna(0)
                                .astype(int)
                                .reset_index()
                            )
                            wide["year"] = wide["year"].astype(str)
                            st.dataframe(wide, use_container_width=True, hide_index=True)

                            st.markdown("**Papers per year assigned to this theme**")
                            denom_display = denom_df.copy()
                            denom_display["year"] = denom_display["year"].astype(str)
                            st.dataframe(denom_display, use_container_width=True, hide_index=True)

with st.sidebar:
    if st.session_state.search_history:
        st.markdown("---")
        st.markdown("##### Recent searches")
        for entry in st.session_state.search_history:
            st.caption(f"**{entry['timestamp']}** — _{entry['query']}_ ({len(entry['results'])} papers)")
