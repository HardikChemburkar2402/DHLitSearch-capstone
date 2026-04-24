# DHLitSearch — Weekly Update (2026-04-23)

## Summary
- Streamlit app remains integrated end-to-end with the backend pipeline (retrieval + evidence-grounded Q&A + topic modeling).
- Implemented professor-requested improvements focused on **researcher usability** (top-5 overview + clearer topics).
- Switched Gemini calls to **Vertex AI-first** (to use the free \(~$300\) credits) with automatic fallback.

## Work completed this week
- **Semantic Search**
  - Added an **“Overview (top 5 papers)”** section that synthesizes the top-ranked abstracts into a single concise paragraph with **clickable inline citations**.
  - Saved the last 10 searches into a **Search History Bank** (session state) for easier backtracking during demos.
- **Theme Explorer / Topic Modeling**
  - Added **human-readable topic labels** generated from each topic’s top keywords (with a keyword-based fallback if the LLM is unavailable).
- **LLM provider**
  - Updated the Gemini integration to **prefer Vertex AI** when `GOOGLE_CLOUD_PROJECT` is set, and show the active provider in the UI.

## Limitations / risks
- **Evaluation constraints:** no human gold question–answer pairs or reference summaries, so metrics are proxy-based and should be treated as indicative.
- **Corpus constraints:** retrieval is limited by the current indexed dataset (coverage/recall depends on what’s in Chroma).
- **Grounding:** answers/overviews are prompted to stay within retrieved abstracts, reducing hallucination risk, but not guaranteeing perfection in all cases.

## Professor feedback status (Prof. Xiao Hu)
- Demo was presented; we are actively implementing requested changes.
- Completed:
  - Top-5 overview synthesis in Semantic Search.
  - Human-readable topic labels.
- In progress / next:
  - Topic modeling workflow: **select prior search → select a topic** (instead of focusing on “number of topics”).
  - Add a visualization showing **top-5 keyword relevance over years** for the selected search/topic.

## How to confirm Vertex AI credits are being used
- In the app sidebar, check the **LLM line**:
  - **“Vertex AI (…project…, …location…)”** indicates requests are billed to Vertex AI.
  - **“AI Studio (API key)”** indicates fallback to API-key mode.

