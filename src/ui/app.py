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

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="K8s RAG Assistant",
    page_icon="☸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS — only for source type badges ──────────────────────────────────
st.markdown("""
<style>
.source-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px 3px 3px 0;
}
.badge-concepts      { background: #dbeafe; color: #1e40af; }
.badge-book          { background: #ede9fe; color: #5b21b6; }
.badge-troubleshoot  { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Knowledge Base")
    st.markdown(
        "Three documents chosen to cover distinct user intents:"
    )

    st.markdown("**🌐 Kubernetes Docs** — `~2,400 chunks`")
    st.caption("kubernetes.io/docs/concepts · *Explain what X is*")

    st.markdown("**📖 Kubernetes in Action** — `~1,200 chunks`")
    st.caption("Marko Lukša, Ch. 1–10 · *How do I do X*")

    st.markdown("**🔧 Troubleshooting Guide** — `~38 chunks`")
    st.caption("learnk8s.io/troubleshooting · *Why isn't X working*")

    st.divider()

    st.markdown("## ⚙️ System Status")

    if st.button("🔍 Check Health", use_container_width=True):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            health = r.json()
            color = "green" if health["status"] == "ok" else "red"
            st.markdown(f"**Status:** :{color}[{health['status']}]")
            col_a, col_b = st.columns(2)
            col_a.markdown(f"{'✅' if health['chroma_ok'] else '❌'} ChromaDB")
            col_b.markdown(f"{'✅' if health['bm25_ok'] else '❌'} BM25")
            with st.expander("Feature Flags"):
                st.json(health["feature_flags"])
        except Exception as e:
            st.error(f"API unreachable: {e}")

    st.divider()

    if st.button("🔄 Reset Session", use_container_width=True):
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

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("☸️ Kubernetes RAG Assistant")
st.caption(
    "Hybrid retrieval (ChromaDB + BM25)  ·  "
    "LangGraph agent  ·  GPT-4o-mini  ·  "
    "Streaming SSE  ·  Session memory"
)
st.divider()

# ── Source badge helper ────────────────────────────────────────────────────────
def _source_html(filename: str) -> str:
    """Return a colored HTML badge based on the source document type."""
    f = filename.lower()
    if "troubleshoot" in f:
        return (
            f'<span class="source-badge badge-troubleshoot">'
            f'🔧 Troubleshooting Guide</span>'
        )
    elif "concepts" in f:
        return (
            f'<span class="source-badge badge-concepts">'
            f'📘 Kubernetes Docs</span>'
        )
    elif "action" in f or f.endswith(".pdf"):
        return (
            f'<span class="source-badge badge-book">'
            f'📖 Kubernetes in Action</span>'
        )
    return f'<span class="source-badge badge-concepts">📄 {filename}</span>'


# ── Example questions (empty state) ───────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    ("🔴", "Why is my Pod stuck in CrashLoopBackOff?"),
    ("🔴", "My Service isn't routing traffic — what should I check?"),
    ("📘", "What's the difference between a Deployment and a StatefulSet?"),
    ("📘", "How does the Kubernetes scheduler decide where to place a Pod?"),
    ("🛠️", "How do I configure CPU and memory limits for a container?"),
    ("🛠️", "How do I perform a rolling update with zero downtime?"),
]

pending = st.session_state.pop("pending_question", None)

# ── Chat history ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    cols = st.columns(2)
    for i, (icon, question) in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            label = f"{icon} {question}"
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()
    st.divider()
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    badges_html = "".join(
                        _source_html(src) for src in msg["sources"]
                    )
                    st.markdown(badges_html, unsafe_allow_html=True)

# ── Chat input ─────────────────────────────────────────────────────────────────
typed_prompt = st.chat_input(
    "Ask about Kubernetes...  e.g. 'Why is my Pod in CrashLoopBackOff?'"
)
prompt = typed_prompt or pending

if prompt:
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
                badges_html = "".join(_source_html(src) for src in sources)
                st.markdown(badges_html, unsafe_allow_html=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "sources": sources,
        }
    )
