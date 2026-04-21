"""
app.py — Main Streamlit entrypoint
Modular single-page UI: auth → sidebar → upload → analysis → chat
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import requests
import sseclient
import streamlit as st

from src.research_intelligence_system.config.settings import settings
from src.research_intelligence_system.constants import LLM_OPTIONS, MAX_POLLS, POLL_INTERVAL

from frontend.styles      import GLOBAL_CSS, THINKING_HTML
from frontend.auth_ui     import render_login_page
from frontend.sidebar_ui  import render_sidebar
from frontend.analysis_ui import render_analysis

BACKEND = settings.BACKEND_ORIGIN_URL

st.set_page_config(
    page_title = "Research AI",
    page_icon  = "⬡",
    layout     = "wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ── Session init ──────────────────────────────────────────────────────────────
def _restore_session(sid: str) -> bool:
    try:
        r = requests.get(f"{BACKEND}/auth/session/{sid}", timeout=5)
        if r.status_code == 200:
            data = r.json()
            st.session_state.token = data["token"]
            st.session_state.user  = data["user"]
            return True
    except Exception:
        pass
    return False


def _init() -> None:
    defaults = {
        "token":           None,
        "user":            None,
        "current_chat_id": None,
        "chats":           {},
        "llm_id":          LLM_OPTIONS[0],
        "allow_search":    False,
        "analysis_cache":  {},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    if not st.session_state.token:
        sid = st.query_params.get("sid")
        if sid:
            _restore_session(sid)


_init()


# ── API helper ────────────────────────────────────────────────────────────────
def _headers() -> Dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def _api(method: str, path: str, show_error: bool = True, **kw):
    try:
        r = getattr(requests, method)(
            f"{BACKEND}{path}", headers=_headers(), timeout=60, **kw
        )
        return r
    except Exception as e:
        if show_error:
            st.error(f"Connection error: {e}")
        return None


# ── Chat management ───────────────────────────────────────────────────────────
def _load_chats() -> None:
    r = _api("get", "/chats", show_error=False)
    if not r or r.status_code != 200:
        return
    for c in r.json():
        cid = c["id"]
        if cid not in st.session_state.chats:
            st.session_state.chats[cid] = {
                "title":    c["title"],
                "messages": [],
                "ready":    False,
                "analyzed": False,
                "llm_id":   c["llm_id"],
            }
        else:
            if c["title"] != "New Chat":
                st.session_state.chats[cid]["title"] = c["title"]

        # restore analyzed state
        chat = st.session_state.chats[cid]
        if not chat.get("ready") or not chat.get("analyzed"):
            r2 = _api("get", f"/chats/{cid}/analysis-status", show_error=False)
            if r2 and r2.status_code == 200:
                status = r2.json().get("status", "")
                if status == "complete":
                    chat["ready"] = chat["analyzed"] = True
                    if cid not in st.session_state.analysis_cache:
                        data = _fetch_analysis(cid)
                        if data:
                            st.session_state.analysis_cache[cid] = data
                elif status in ("running", "pending", "ready"):
                    chat["ready"] = True


def _new_chat() -> None:
    r = _api("post", "/chats", json={
        "title":        "New Chat",
        "llm_id":       st.session_state.llm_id,
        "allow_search": st.session_state.allow_search,
    })
    if r and r.status_code == 201:
        data = r.json()
        cid  = data["id"]
        st.session_state.chats[cid] = {
            "title": data["title"], "messages": [],
            "ready": False, "analyzed": False, "llm_id": data["llm_id"],
        }
        st.session_state.current_chat_id = cid
        st.rerun()


def _delete_chat(cid: str) -> None:
    _api("delete", f"/chats/{cid}")
    st.session_state.chats.pop(cid, None)
    st.session_state.analysis_cache.pop(cid, None)
    if st.session_state.current_chat_id == cid:
        st.session_state.current_chat_id = None
    st.rerun()


def _switch_chat(cid: str) -> None:
    st.session_state.current_chat_id = cid
    chat = st.session_state.chats[cid]
    if not chat["messages"]:
        r = _api("get", f"/chats/{cid}/history")
        if r and r.status_code == 200:
            chat["messages"] = r.json()
    st.rerun()


# ── Upload & analysis ─────────────────────────────────────────────────────────
def _upload(cid: str, file) -> bool:
    r = _api("post", f"/chats/{cid}/upload",
             files={"file": (file.name, file.getvalue(), "application/pdf")})
    return r is not None and r.status_code == 200


def _poll_ingest(cid: str) -> str:
    bar = st.progress(0, text="Processing PDF …")
    i   = 0
    while True:
        time.sleep(POLL_INTERVAL)
        try:
            r = _api("get", f"/chats/{cid}/status", show_error=False)
            s = r.json().get("status", "unknown") if r else "unknown"
        except Exception:
            s = "unknown"
        i += 1
        bar.progress(min(int(i / MAX_POLLS * 100), 95), text=f"⏳ {s} ({i * POLL_INTERVAL:.0f}s)")
        if s == "ready":
            bar.progress(100, text="✅ Indexed!")
            return "ready"
        if s == "failed":
            bar.empty()
            return "failed"
        if i > MAX_POLLS * 3:
            bar.empty()
            return "timeout"


def _trigger_analysis(cid: str) -> bool:
    r = _api("post", f"/chats/{cid}/analyze", show_error=True)
    if not r or r.status_code != 200:
        st.error("Failed to start analysis.")
        return False

    bar = st.progress(0, text="🤖 Running agents …")
    i   = 0
    while True:
        time.sleep(3)
        try:
            r = _api("get", f"/chats/{cid}/analysis-status", show_error=False)
            s = r.json().get("status", "unknown") if r else "unknown"
        except Exception:
            s = "unknown"
        i += 1
        bar.progress(min(int(i / 30 * 100), 95), text=f"🤖 {s} ({i * 3}s)")
        if s == "complete":
            bar.progress(100, text="✅ Analysis complete!")
            return True
        if s == "failed":
            bar.empty()
            st.error("Analysis failed.")
            return False
        if i > 80:
            bar.empty()
            st.warning("Taking longer than expected — check back soon.")
            return False


def _fetch_analysis(cid: str) -> Optional[Dict]:
    r = _api("get", f"/chats/{cid}/analysis", show_error=False)
    if r and r.status_code == 200:
        return r.json()
    return None


# ── Streaming ─────────────────────────────────────────────────────────────────
def _fix_format(text: str) -> str:
    import re
    text = re.sub(r'(\S)\s{0,2}(\d+\.\s+\*\*)', r'\1\n\n\2', text)
    text = re.sub(r'(\S)\s{0,2}(\*\*Summary)',   r'\1\n\n\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _stream_answer(cid: str, message: str, llm_id: str, allow_search: bool) -> str:
    placeholder = st.empty()
    placeholder.markdown(THINKING_HTML, unsafe_allow_html=True)
    full = ""
    try:
        with requests.post(
            f"{BACKEND}/chats/{cid}/stream",
            headers=_headers(),
            json={"message": message, "llm_id": llm_id, "allow_search": allow_search},
            stream=True, timeout=120,
        ) as resp:
            for event in sseclient.SSEClient(resp).events():
                if event.data == "[DONE]":
                    break
                if event.data.startswith("[ERROR]"):
                    full = event.data
                    break
                full += event.data.replace(chr(160), " ")
                placeholder.markdown(full + "▌", unsafe_allow_html=True)
            full = _fix_format(full)
        placeholder.markdown(full, unsafe_allow_html=True)
        return full
    except Exception:
        r      = _api("post", f"/chats/{cid}/message",
                      json={"message": message, "llm_id": llm_id,
                            "allow_search": allow_search})
        answer = r.json().get("answer", "Error.") if r else "Error."
        placeholder.markdown(answer, unsafe_allow_html=True)
        return answer


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.token:
    render_login_page(BACKEND)
    st.stop()

_load_chats()

# ── Sidebar ───────────────────────────────────────────────────────────────────
action = render_sidebar(BACKEND, LLM_OPTIONS, _api)
if action == "__new__":
    _new_chat()
elif action and action.startswith("__select__"):
    _switch_chat(action.replace("__select__", ""))
elif action and action.startswith("__delete__"):
    _delete_chat(action.replace("__delete__", ""))

# ── Landing ───────────────────────────────────────────────────────────────────
if not st.session_state.current_chat_id:
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;
                justify-content:center;min-height:60vh;text-align:center'>
        <div style='font-family:Space Mono,monospace;font-size:3rem;
                    font-weight:700;letter-spacing:-0.04em;
                    background:linear-gradient(135deg,#6366f1,#a5b4fc,#10b981);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    margin-bottom:16px'>
            Research AI
        </div>
        <div style='color:#8888aa;font-size:1rem;max-width:480px;line-height:1.7;
                    margin-bottom:32px'>
            Upload scientific papers and get instant AI-powered analysis —
            summaries, entities, research gaps, knowledge graphs, and more.
        </div>
        <div style='color:#55556a;font-size:0.82rem'>
            ← Create a new chat to get started
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

cid  = st.session_state.current_chat_id
chat = st.session_state.chats[cid]

# ── Chat title ────────────────────────────────────────────────────────────────
st.markdown(
    f"<h1 style='font-family:Space Mono,monospace;font-size:1.4rem;"
    f"font-weight:700;letter-spacing:-0.03em;margin-bottom:4px'>"
    f"{chat['title']}</h1>",
    unsafe_allow_html=True,
)

# ── PDF Upload ────────────────────────────────────────────────────────────────
with st.expander("📎 Upload PDF(s)", expanded=not chat["ready"]):
    files = st.file_uploader(
        "Drop research papers here",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"up_{cid}",
        label_visibility="collapsed",
    )
    if files and st.button("Upload & Process", type="primary"):
        for f in files:
            with st.spinner(f"Uploading {f.name} …"):
                ok = _upload(cid, f)
            if not ok:
                st.error(f"Upload failed: {f.name}")
                continue
            status = _poll_ingest(cid)
            if status == "ready":
                chat["ready"] = True
                st.success(f"✅ {f.name} indexed successfully!")
            elif status == "failed":
                st.error(f"❌ Indexing failed: {f.name}")
            else:
                chat["ready"] = True
                st.warning("Still processing — you can continue.")
        st.rerun()

if not chat["ready"]:
    st.markdown(
        "<div style='text-align:center;padding:40px;color:#55556a;"
        "font-size:0.9rem'>Upload at least one PDF to begin.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Get Analysis ──────────────────────────────────────────────────────────────
if not chat.get("analyzed"):
    st.markdown(
        '<div class="analysis-banner">'
        '✅ Paper ready — run the full multi-agent analysis pipeline to extract '
        'entities, summaries, research gaps, knowledge graph, comparison, and literature review.'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.button("⬡ Run Analysis", type="primary", use_container_width=True):
        success = _trigger_analysis(cid)
        if success:
            chat["analyzed"] = True
            data = _fetch_analysis(cid)
            if data:
                st.session_state.analysis_cache[cid] = data
        st.rerun()

# ── Analysis results ──────────────────────────────────────────────────────────
if chat.get("analyzed"):
    analysis = st.session_state.analysis_cache.get(cid)
    if not analysis:
        analysis = _fetch_analysis(cid)
        if analysis:
            st.session_state.analysis_cache[cid] = analysis
    if analysis:
        render_analysis(analysis)

# ── Chat history ──────────────────────────────────────────────────────────────
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and msg.get("source"):
            st.caption(
                f"📌 {msg['source']} "
                f"· 🎯 confidence {msg.get('confidence', '')}"
            )

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about your research paper …")

if user_input:
    chat["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        answer = _stream_answer(
            cid, user_input,
            st.session_state.llm_id,
            st.session_state.allow_search,
        )

    chat["messages"].append({
        "role": "assistant", "content": answer, "source": "stream"
    })
    st.rerun()