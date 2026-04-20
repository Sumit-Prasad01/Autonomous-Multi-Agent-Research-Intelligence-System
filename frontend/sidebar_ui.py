"""sidebar_ui.py — Sidebar rendering"""
from __future__ import annotations
import requests
import streamlit as st
from typing import Dict, Optional


def render_sidebar(
    backend:   str,
    llm_options: list,
    api_fn,
) -> Optional[str]:
    """
    Render sidebar. Returns chat_id to delete if delete button pressed,
    else None.
    """
    with st.sidebar:
        # ── Brand ─────────────────────────────────────────────────────────
        st.markdown("""
        <div style='padding:16px 0 8px'>
            <div style='font-family:Space Mono,monospace;font-size:1.1rem;
                        font-weight:700;letter-spacing:-0.03em;
                        color:#f0f0f8'>
                ⬡ Research AI
            </div>
            <div style='font-size:0.72rem;color:#55556a;
                        letter-spacing:0.06em;text-transform:uppercase;
                        margin-top:2px'>
                Multi-Agent Intelligence
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"<div style='font-size:0.8rem;color:#8888aa;padding:0 0 12px'>"
            f"Signed in as <b style='color:#a5b4fc'>"
            f"{st.session_state.user}</b></div>",
            unsafe_allow_html=True,
        )

        # ── Settings ───────────────────────────────────────────────────────
        st.selectbox("Language Model", llm_options, key="llm_id",
                     label_visibility="collapsed")
        st.toggle("🌐 Web Search", key="allow_search")
        st.divider()

        # ── Actions ────────────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            if st.button("＋ New Chat", use_container_width=True):
                return "__new__"
        with col2:
            if st.button("Sign Out", use_container_width=True):
                sid = st.query_params.get("sid")
                if sid:
                    requests.delete(f"{backend}/auth/session/{sid}", timeout=5)
                api_fn("post", "/auth/logout", show_error=False)
                st.session_state.update({
                    "token": None, "user": None,
                    "chats": {}, "current_chat_id": None,
                    "analysis_cache": {},
                })
                st.query_params.clear()
                st.rerun()

        st.divider()

        # ── Chat list ──────────────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:0.72rem;color:#55556a;"
            "letter-spacing:0.08em;text-transform:uppercase;"
            "margin-bottom:8px'>Your Chats</div>",
            unsafe_allow_html=True,
        )

        to_delete = None
        for cid, chat in list(st.session_state.chats.items()):
            is_active  = cid == st.session_state.current_chat_id
            is_analyzed = chat.get("analyzed", False)
            is_ready    = chat.get("ready", False)

            status_dot = (
                "🟢" if is_active else
                "✅" if is_analyzed else
                "📄" if is_ready else
                "○"
            )
            title = chat["title"][:30] + ("…" if len(chat["title"]) > 30 else "")

            cols = st.columns([0.82, 0.18])
            btn_style = (
                "background:rgba(99,102,241,0.12);border-color:rgba(99,102,241,0.3);"
                if is_active else ""
            )
            if cols[0].button(
                f"{status_dot} {title}",
                key=f"sel_{cid}",
                use_container_width=True,
            ):
                return f"__select__{cid}"
            if cols[1].button("✕", key=f"del_{cid}", help="Delete chat"):
                to_delete = cid

        if to_delete:
            return f"__delete__{to_delete}"

    return None