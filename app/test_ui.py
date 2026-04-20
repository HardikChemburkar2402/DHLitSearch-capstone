import streamlit as st
import time
import os

# Page Config
st.set_page_config(
    page_title="DHLitSearch",
    page_icon="🧬",
    layout="wide",
)

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css(os.path.join(os.path.dirname(__file__), 'style.css'))
except FileNotFoundError:
    pass

st.title("🧬 DHLitSearch")
st.markdown("### NLP-powered biomedical literature analysis tool")

# Ethics Disclaimer
st.markdown("""
<div class="ethics-disclaimer">
    <strong>⚠️ Ethics Disclaimer:</strong> This tool is for research assistance strictly, and is not designed for clinical decision support.
</div>
""", unsafe_allow_html=True)

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about Digital Health Literacy..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Synthesizing biomedical research..."):
            time.sleep(2)  # simulate RAG
            response = "**Source 1 [Paper ID: 1284]:**\nDigital health literacy is defined as the ability to seek, find, understand, and appraise health information from electronic sources."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
