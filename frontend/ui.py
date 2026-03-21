from __future__ import annotations

import time
from typing import Dict, List, Optional

import requests
import sseclient
import streamlit as st

from src.research_intelligence_system.config.settings import settings
from src.research_intelligence_system.constants import LLM_OPTIONS, MAX_POLLS, POLL_INTERVAL

BACKEND = settings.BACKEND_ORIGIN_URL

st.set_page_config(
    page_title="Research AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.badge{font-size:0.78rem;color:#888;margin-bottom:6px}
@keyframes thinking-dot{0%,80%,100%{transform:scale(.6);opacity:.4}40%{transform:scale(1);opacity:1}}
.thinking-wrap{display:flex;align-items:center;gap:8px;padding:6px 0;color:#888;font-size:.95rem;font-style:italic}
.thinking-dots{display:flex;gap:4px}
.thinking-dots span{display:inline-block;width:7px;height:7px;border-radius:50%;background:#888;animation:thinking-dot 1.2s infinite ease-in-out}
.thinking-dots span:nth-child(2){animation-delay:.2s}
.thinking-dots span:nth-child(3){animation-delay:.4s}
</style>
""", unsafe_allow_html=True)

THINKING_HTML = """
<div class="thinking-wrap">
    <span>Thinking</span>
    <div class="thinking-dots"><span></span><span></span><span></span></div>
</div>
"""


# ── Session helpers ───────────────────────────────────────────────────────────
def _restore_session_from_server(session_id: str) -> bool:
    """Call backend to get token from session_id stored in Redis."""
    try:
        r = requests.get(f"{BACKEND}/auth/session/{session_id}", timeout=5)
        if r.status_code == 200:
            data = r.json()
            st.session_state.token = data["token"]
            st.session_state.user  = data["user"]
            return True
    except Exception:
        pass
    return False


# ── Session init ──────────────────────────────────────────────────────────────
def _init():
    for k, v in {
        "token":           None,
        "user":            None,
        "current_chat_id": None,
        "chats":           {},
        "llm_id":          LLM_OPTIONS[0],
        "allow_search":    False,
    }.items():
        st.session_state.setdefault(k, v)

    # restore session from Redis via session_id in query params
    if not st.session_state.token:
        sid = st.query_params.get("sid")
        if sid:
            _restore_session_from_server(sid)

_init()


# ── API ───────────────────────────────────────────────────────────────────────
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


# ── Auth page ─────────────────────────────────────────────────────────────────
def _login_page():
    st.title("🔬 Research AI")
    tab_login, tab_reg = st.tabs(["Login", "Register"])

    with tab_login:
        email    = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", type="primary", use_container_width=True):
            r = requests.post(
                f"{BACKEND}/auth/login",
                json={"email": email, "password": password}, timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                st.session_state.token = data["access_token"]
                st.session_state.user  = email
                # store only session_id in URL — token stays in Redis
                st.query_params["sid"] = data["session_id"]
                st.rerun()
            else:
                st.error("Invalid credentials.")

    with tab_reg:
        r_email    = st.text_input("Email",    key="reg_email")
        r_username = st.text_input("Username", key="reg_user")
        r_password = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Register", type="primary", use_container_width=True):
            r = requests.post(
                f"{BACKEND}/auth/register",
                json={"email": r_email, "username": r_username, "password": r_password},
                timeout=10,
            )
            if r.status_code == 201:
                st.success("Registered! Please login.")
            else:
                st.error(r.json().get("detail", "Registration failed."))


# ── Chat management ───────────────────────────────────────────────────────────
def _load_chats():
    r = _api("get", "/chats", show_error=False)
    if r and r.status_code == 200:
        for c in r.json():
            cid = c["id"]
            if cid not in st.session_state.chats:
                st.session_state.chats[cid] = {
                    "title":    c["title"],
                    "messages": [],
                    "ready":    False,
                    "llm_id":   c["llm_id"],
                }
            else:
                # only sync if backend title is not default
                if c["title"] != "New Chat":
                    st.session_state.chats[cid]["title"] = c["title"]


def _new_chat():
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
            "ready": False, "llm_id": data["llm_id"],
        }
        st.session_state.current_chat_id = cid
        st.rerun()


def _delete_chat(cid: str):
    _api("delete", f"/chats/{cid}")
    st.session_state.chats.pop(cid, None)
    if st.session_state.current_chat_id == cid:
        st.session_state.current_chat_id = None
    st.rerun()


def _switch_chat(cid: str):
    st.session_state.current_chat_id = cid
    chat = st.session_state.chats[cid]
    if not chat["messages"]:
        r = _api("get", f"/chats/{cid}/history")
        if r and r.status_code == 200:
            chat["messages"] = r.json()
            chat["ready"]    = bool(chat["messages"])
    st.rerun()


# ── Upload ────────────────────────────────────────────────────────────────────
def _upload(cid: str, file) -> bool:
    r = _api("post", f"/chats/{cid}/upload",
             files={"file": (file.name, file.getvalue(), "application/pdf")})
    return r is not None and r.status_code == 200


def _poll_status(cid: str) -> str:
    bar = st.progress(0, text="⏳ Processing …")
    i   = 0
    while True:
        time.sleep(POLL_INTERVAL)
        try:
            r = _api("get", f"/chats/{cid}/status", show_error=False)
            s = r.json().get("status", "unknown") if r else "unknown"
        except Exception:
            s = "unknown"
        i += 1
        pct = min(int(i / MAX_POLLS * 100), 95)
        bar.progress(pct, text=f"⏳ {s} … ({i * POLL_INTERVAL:.0f}s)")
        if s == "ready":
            bar.progress(100, text="✅ Ready!")
            return "ready"
        if s == "failed":
            bar.empty()
            return "failed"
        if i >= MAX_POLLS:
            bar.progress(95, text="⏳ Still processing … please wait")
            if i > MAX_POLLS * 3:
                bar.empty()
                return "timeout"


# ── Streaming ─────────────────────────────────────────────────────────────────
def _stream_answer(cid: str, message: str, llm_id: str, allow_search: bool) -> str:
    placeholder = st.empty()
    placeholder.markdown(THINKING_HTML, unsafe_allow_html=True)
    full = ""
    try:
        with requests.post(
            f"{BACKEND}/chats/{cid}/stream",
            headers=_headers(),
            json={"message": message, "llm_id": llm_id, "allow_search": allow_search},
            stream=True, timeout=90,
        ) as resp:
            for event in sseclient.SSEClient(resp).events():
                if event.data == "[DONE]": break
                if event.data.startswith("[ERROR]"):
                    full = event.data; break
                full += event.data
                placeholder.markdown(full + "▌", unsafe_allow_html=True)
        placeholder.markdown(full, unsafe_allow_html=True)
        return full
    except Exception:
        r      = _api("post", f"/chats/{cid}/message",
                      json={"message": message, "llm_id": llm_id, "allow_search": allow_search})
        answer = r.json().get("answer", "Error.") if r else "Error."
        placeholder.markdown(answer, unsafe_allow_html=True)
        return answer


# ── Gate: require login ───────────────────────────────────────────────────────
if not st.session_state.token:
    _login_page()
    st.stop()

_load_chats()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research AI")
    st.caption(f"Logged in as **{st.session_state.user}**")

    if st.button("Logout", use_container_width=True):
        # delete session from Redis
        sid = st.query_params.get("sid")
        if sid:
            requests.delete(f"{BACKEND}/auth/session/{sid}", timeout=5)
        _api("post", "/auth/logout", show_error=False)
        st.session_state.update({
            "token": None, "user": None,
            "chats": {}, "current_chat_id": None,
        })
        st.query_params.clear()
        st.rerun()

    st.divider()
    st.selectbox("Model", LLM_OPTIONS, key="llm_id")
    st.toggle("🌐 Web Search", key="allow_search")
    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        _new_chat()

    st.markdown("**Your Chats**")
    to_delete = None
    for cid, chat in list(st.session_state.chats.items()):
        is_active = cid == st.session_state.current_chat_id
        cols  = st.columns([0.78, 0.22])
        label = ("🟢 " if is_active else "") + ("✅ " if chat["ready"] else "") + chat["title"]
        if cols[0].button(label, key=f"sel_{cid}", use_container_width=True):
            _switch_chat(cid)
        if cols[1].button("🗑", key=f"del_{cid}"):
            to_delete = cid
    if to_delete:
        _delete_chat(to_delete)


# ── Landing ───────────────────────────────────────────────────────────────────
if not st.session_state.current_chat_id:
    st.title("📄 Research Intelligence System")
    st.markdown("Create a new chat from the sidebar to get started.")
    st.stop()

cid  = st.session_state.current_chat_id
chat = st.session_state.chats[cid]
st.title(chat["title"])

# ── PDF upload ────────────────────────────────────────────────────────────────
with st.expander("📎 Upload PDF(s)", expanded=not chat["ready"]):
    files = st.file_uploader(
        "Drop research papers here", type=["pdf"],
        accept_multiple_files=True, key=f"up_{cid}",
    )
    if files and st.button("Upload & Process", type="primary"):
        for f in files:
            with st.spinner(f"Uploading {f.name} …"):
                ok = _upload(cid, f)
            if not ok:
                st.error(f"Upload failed: {f.name}")
                continue
            status = _poll_status(cid)
            if status == "ready":
                chat["ready"] = True
                st.success(f"✅ {f.name} indexed!")
                st.rerun()    # ← rerun immediately so chat input appears
            elif status == "failed":
                st.error(f"❌ Indexing failed for {f.name}")
            else:
                chat["ready"] = True
                st.warning("⏱ Processing may still be running — refreshing …")
                st.rerun()    # ← rerun on timeout too

if not chat["ready"]:
    st.info("Upload at least one PDF to start chatting.")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and msg.get("source"):
            st.caption(f'📌 {msg["source"]} · 🎯 {msg.get("confidence", "")}')

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about your research paper …")

if user_input:
    if not chat["messages"]:
        chat["title"] = user_input[:40]
        # immediately update sidebar title in session state
        st.session_state.chats[cid]["title"] = user_input[:40]

    chat["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        answer = _stream_answer(
            cid, user_input,
            st.session_state.llm_id,
            st.session_state.allow_search,
        )

    chat["messages"].append({"role": "assistant", "content": answer, "source": "stream"})
    st.rerun()
