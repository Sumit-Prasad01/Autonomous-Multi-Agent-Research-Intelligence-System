from __future__ import annotations

import time
import uuid
from typing import Dict, List

import requests
import sseclient
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND       = "http://127.0.0.1:8000/api/v1"
POLL_INTERVAL = 1.5
MAX_POLLS     = 40

LLM_OPTIONS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
]

st.set_page_config(
    page_title="Research AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.badge { font-size: 0.78rem; color: #888; margin-bottom: 6px; }
.stChatMessage { border-radius: 10px; }
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}
.cursor {
    display: inline-block;
    width: 0.55em; height: 1.1em;
    background: currentColor;
    vertical-align: text-bottom;
    margin-left: 1px; border-radius: 1px;
    animation: blink 0.8s step-start infinite;
}
@keyframes thinking-dot {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40%            { transform: scale(1);   opacity: 1;   }
}
.thinking-wrap {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 0; color: #888;
    font-size: 0.95rem; font-style: italic;
}
.thinking-dots { display: flex; gap: 4px; }
.thinking-dots span {
    display: inline-block; width: 7px; height: 7px;
    border-radius: 50%; background: #888;
    animation: thinking-dot 1.2s infinite ease-in-out;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
</style>
""", unsafe_allow_html=True)

THINKING_HTML = """
<div class="thinking-wrap">
    <span>Thinking</span>
    <div class="thinking-dots"><span></span><span></span><span></span></div>
</div>
"""


# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    for k, v in {
        "chats": {}, "current_chat_id": None,
        "llm_id": LLM_OPTIONS[0], "allow_search": False,
    }.items():
        st.session_state.setdefault(k, v)

_init()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _new_chat() -> str:
    cid = str(uuid.uuid4())
    st.session_state.chats[cid] = {"title": "New Chat", "messages": [], "pdfs": [], "ready": False}
    st.session_state.current_chat_id = cid
    return cid

def _delete_chat(cid: str):
    st.session_state.chats.pop(cid, None)
    if st.session_state.current_chat_id == cid:
        st.session_state.current_chat_id = None
    st.rerun()

def _chat() -> Dict:
    return st.session_state.chats[st.session_state.current_chat_id]


# ── API ───────────────────────────────────────────────────────────────────────
def _upload(cid: str, file) -> bool:
    try:
        r = requests.post(
            f"{BACKEND}/upload",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            data={"chat_id": cid}, timeout=120,
        )
        return r.status_code == 200
    except Exception as e:
        st.error(f"Upload error: {e}")
        return False

def _poll_status(cid: str) -> str:
    bar = st.progress(0, text="⏳ Processing PDF …")
    for i in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        try:
            s = requests.get(f"{BACKEND}/status/{cid}", timeout=5).json().get("status", "unknown")
        except Exception:
            s = "unknown"
        bar.progress(min(int((i + 1) / MAX_POLLS * 100), 95), text=f"⏳ {s} …")
        if s == "ready":  bar.progress(100, text="✅ Ready!"); return "ready"
        if s == "failed": bar.empty(); return "failed"
    bar.empty()
    return "timeout"

def _ask(cid: str, messages: List[Dict], allow_search: bool) -> Dict:
    try:
        r = requests.post(
            f"{BACKEND}/chat",
            json={"chat_id": cid, "llm_id": st.session_state.llm_id,
                  "allow_search": allow_search, "messages": messages},
            timeout=60,
        )
        return r.json() if r.status_code == 200 else {
            "answer": f"Error {r.status_code}: {r.text}", "source": "error", "confidence": 0.0}
    except Exception as e:
        return {"answer": f"⚠️ {e}", "source": "error", "confidence": 0.0}

def _stream_answer(cid: str, messages: List[Dict], allow_search: bool) -> str:
    placeholder = st.empty()
    placeholder.markdown(THINKING_HTML, unsafe_allow_html=True)

    full = ""
    try:
        with requests.post(
            f"{BACKEND}/chat/stream",
            json={"chat_id": cid, "llm_id": st.session_state.llm_id,
                  "allow_search": allow_search, "messages": messages},
            stream=True, timeout=90,
        ) as resp:
            for event in sseclient.SSEClient(resp).events():
                if event.data == "[DONE]":
                    break
                if event.data.startswith("[ERROR]"):
                    full = event.data
                    break
                full += event.data
                # ── KEY FIX: unsafe_allow_html=True so ** renders as bold ────
                placeholder.markdown(full + "▌", unsafe_allow_html=True)

        placeholder.markdown(full, unsafe_allow_html=True)
        return full

    except Exception:
        placeholder.markdown(THINKING_HTML, unsafe_allow_html=True)
        answer = _ask(cid, messages, allow_search).get("answer", "")
        placeholder.markdown(answer, unsafe_allow_html=True)
        return answer


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research AI")
    st.selectbox("Model", LLM_OPTIONS, key="llm_id")
    st.toggle("🌐 Web Search", key="allow_search")
    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        _new_chat(); st.rerun()

    st.markdown("**Chats**")
    to_delete = None
    for cid, chat in st.session_state.chats.items():
        cols = st.columns([0.82, 0.18])
        label = ("✅ " if chat["ready"] else "⏳ ") + chat["title"]
        if cols[0].button(label, key=f"sel_{cid}", use_container_width=True):
            st.session_state.current_chat_id = cid; st.rerun()
        if cols[1].button("🗑", key=f"del_{cid}"):
            to_delete = cid
    if to_delete:
        _delete_chat(to_delete)


# ── Landing ───────────────────────────────────────────────────────────────────
if not st.session_state.current_chat_id:
    st.title("📄 Research Intelligence System")
    st.markdown("Create a new chat from the sidebar, then upload a research paper.")
    st.stop()

cid  = st.session_state.current_chat_id
chat = _chat()
st.title(chat["title"])

# ── PDF upload ────────────────────────────────────────────────────────────────
with st.expander("📎 Upload PDF(s)", expanded=not chat["ready"]):
    files = st.file_uploader(
        "Drop research papers here", type=["pdf"],
        accept_multiple_files=True, key=f"uploader_{cid}",
    )
    if files and st.button("Upload & Process", type="primary"):
        for f in files:
            if f.name in chat["pdfs"]:
                st.info(f"{f.name} already uploaded — skipping."); continue
            with st.spinner(f"Uploading {f.name} …"):
                ok = _upload(cid, f)
            if not ok:
                st.error(f"Upload failed: {f.name}"); continue
            chat["pdfs"].append(f.name)
            status = _poll_status(cid)
            if status == "ready":
                chat["ready"] = True; st.success(f"✅ {f.name} indexed!")
            elif status == "failed":
                st.error(f"❌ Indexing failed for {f.name}")
            else:
                st.warning(f"⏱ Timed out for {f.name} — try asking anyway.")
        st.rerun()

if not chat["ready"]:
    st.info("Upload at least one PDF to start chatting.")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        # ── KEY FIX: unsafe_allow_html=True on all message renders ───────────
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and msg.get("meta"):
            m = msg["meta"]
            cached_tag = " · ⚡ cached" if m.get("cached") else ""
            st.caption(f'📌 {m["source"]} · 🎯 {round(m["confidence"], 2)}{cached_tag}')

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about your research paper …")

if user_input:
    if not chat["messages"]:
        chat["title"] = user_input[:40]

    chat["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    api_messages = [{"role": m["role"], "content": m["content"]} for m in chat["messages"]]

    with st.chat_message("assistant"):
        answer = _stream_answer(cid, api_messages, st.session_state.allow_search)

    meta_resp = _ask(cid, api_messages, st.session_state.allow_search)

    chat["messages"].append({
        "role": "assistant", "content": answer,
        "meta": {
            "source":     meta_resp.get("source", "unknown"),
            "confidence": meta_resp.get("confidence", 0.0),
            "cached":     meta_resp.get("cached", False),
        },
    })
    st.rerun()
