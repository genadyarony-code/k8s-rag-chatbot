"""
Streamlit chat UI. Sends questions to the FastAPI backend and displays the response
token by token as it streams in.

The sidebar has a health check button and a session reset button. Source citations
appear in a collapsible expander after each assistant reply.

API_URL defaults to http://localhost:8000 and is overridden by the API_URL
environment variable when running inside Docker Compose.
"""

import os
import json
import uuid

import requests
import streamlit as st

# Target FastAPI backend — overridden via env var in Docker Compose
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="K8s Assistant",
    page_icon="☸️",
    layout="wide",
)
st.title("☸️ Kubernetes RAG Assistant")
st.caption("Powered by LangGraph + ChromaDB + GPT-4o-mini")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("System Status")

    if st.button("🔍 Check Health"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            health = r.json()
            color = "green" if health["status"] == "ok" else "red"
            st.markdown(f"**Status:** :{color}[{health['status']}]")
            st.markdown(f"**ChromaDB:** {'✅' if health['chroma_ok'] else '❌'}")
            st.markdown(f"**BM25:** {'✅' if health['bm25_ok'] else '❌'}")
            with st.expander("Feature Flags"):
                st.json(health["feature_flags"])
        except Exception as e:
            st.error(f"API unreachable: {e}")

    st.divider()

    if st.button("🔄 Reset Session"):
        try:
            requests.post(
                f"{API_URL}/reset/{st.session_state.session_id}", timeout=5
            )
        except Exception:
            pass
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.text(f"📄 {src}")

if prompt := st.chat_input("Ask about Kubernetes... e.g. 'Why is my Pod in CrashLoopBackOff?'"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        sources = []
        error_occurred = False

        # Show a waiting indicator until the first token arrives from the LLM.
        # Without this, the chat bubble appears empty for 1-2 seconds while the
        # backend runs embedding + ChromaDB query + waits for GPT to start generating,
        # which feels like the app has frozen. Once the first token arrives the
        # placeholder is replaced by the streaming text automatically.
        placeholder.markdown("*⏳ Searching documentation and generating answer...*")

        try:
            with requests.post(
                f"{API_URL}/chat",
                json={
                    "question": prompt,
                    "session_id": st.session_state.session_id,
                },
                stream=True,
                timeout=60,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue
                    if not line.startswith(b"data: "):
                        continue

                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    if data.get("done"):
                        sources = data.get("sources", [])
                    else:
                        token = data.get("token", "")
                        full_response += token
                        placeholder.markdown(full_response + "▌")

        except requests.exceptions.Timeout:
            full_response = "⏱️ Request timed out. Please try again."
            error_occurred = True
        except Exception as e:
            full_response = f"❌ Error: {e}"
            error_occurred = True

        placeholder.markdown(full_response)

        if sources and not error_occurred:
            with st.expander("📚 Sources"):
                for src in sources:
                    st.text(f"📄 {src}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "sources": sources,
        }
    )
