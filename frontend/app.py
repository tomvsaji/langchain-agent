"""Streamlit chat frontend with session management and knowledge base UI."""

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="LangChain Agent", page_icon="🤖", layout="wide")

# ── helpers ──────────────────────────────────────────────────────────────────


def api(method: str, path: str, **kwargs):
    """Call the FastAPI backend. Returns parsed JSON or raises."""
    url = f"{API_BASE_URL}{path}"
    resp = getattr(requests, method)(url, timeout=15, **kwargs)
    resp.raise_for_status()
    return resp.json()


# ── session state defaults ───────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── sidebar: sessions & knowledge base ───────────────────────────────────────

with st.sidebar:
    st.header("Chat Sessions")

    if st.button("New Chat", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

    try:
        sessions = api("get", "/sessions")
    except requests.RequestException:
        sessions = []
        st.warning("Backend unavailable")

    for sess in sessions:
        label = f"{sess['title'][:30]}"
        if st.button(label, key=sess["id"], use_container_width=True):
            st.session_state.session_id = sess["id"]
            try:
                msgs = api("get", f"/sessions/{sess['id']}/messages")
                st.session_state.messages = [
                    {"role": m["role"], "content": m["content"]} for m in msgs
                ]
            except requests.RequestException:
                st.session_state.messages = []
            st.rerun()

    st.divider()
    st.header("Knowledge Base")

    doc_text = st.text_area("Add a document", placeholder="Paste text here...")
    if st.button("Add Document", use_container_width=True) and doc_text.strip():
        try:
            result = api("post", "/documents", json={"texts": [doc_text]})
            st.success(f"Added {result['count']} chunk(s)")
        except requests.RequestException as exc:
            st.error(f"Failed: {exc}")

    if st.button("Seed Sample Docs", use_container_width=True):
        try:
            result = api("post", "/documents/seed")
            st.success(f"Seeded {result['seeded']} chunks")
        except requests.RequestException as exc:
            st.error(f"Failed: {exc}")

    try:
        count_data = api("get", "/documents/count")
        st.caption(f"Documents in store: {count_data['count']}")
    except requests.RequestException:
        st.caption("Documents in store: unavailable")

# ── main chat area ───────────────────────────────────────────────────────────

st.title("LangChain Agent Chat")
st.caption("Powered by ChromaDB vector memory and SQLite chat history")

# Display existing messages
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "human" else "assistant"
    with st.chat_message(role):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "human", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {"message": prompt}
                if st.session_state.session_id:
                    payload["session_id"] = st.session_state.session_id
                data = api("post", "/chat", json=payload)
                answer = data["response"]
                st.session_state.session_id = data["session_id"]
            except requests.RequestException as exc:
                answer = f"Error: {exc}"
        st.write(answer)
    st.session_state.messages.append({"role": "ai", "content": answer})
