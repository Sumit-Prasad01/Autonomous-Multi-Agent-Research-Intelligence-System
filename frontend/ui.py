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
@keyframes thinking-dot{0%,80%,100%{transform:scale(.6);opacity:.4}40%{transform:scale(1);opacity:1}}
.thinking-wrap{display:flex;align-items:center;gap:8px;padding:6px 0;color:#888;font-size:.95rem;font-style:italic}
.thinking-dots{display:flex;gap:4px}
.thinking-dots span{display:inline-block;width:7px;height:7px;border-radius:50%;background:#888;animation:thinking-dot 1.2s infinite ease-in-out}
.thinking-dots span:nth-child(2){animation-delay:.2s}
.thinking-dots span:nth-child(3){animation-delay:.4s}
.entity-tag{display:inline-block;background:#1d3a5f;color:#60a5fa;padding:2px 8px;border-radius:12px;font-size:0.8rem;margin:2px}
.gap-item{border-left:3px solid #7c3aed;padding:6px 10px;margin:4px 0;border-radius:0 4px 4px 0;background:#1a1a2e}
.analysis-banner{background:#0f3460;border-radius:8px;padding:12px 16px;margin:8px 0;border:1px solid #1a5276}
</style>
""", unsafe_allow_html=True)

THINKING_HTML = """
<div class="thinking-wrap">
    <span>Thinking</span>
    <div class="thinking-dots"><span></span><span></span><span></span></div>
</div>
"""


# ── Session init ──────────────────────────────────────────────────────────────
def _restore_session(session_id: str) -> bool:
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


def _init():
    for k, v in {
        "token":           None,
        "user":            None,
        "current_chat_id": None,
        "chats":           {},
        "llm_id":          LLM_OPTIONS[0],
        "allow_search":    False,
        "analysis_cache":  {},   # cid → analysis data
    }.items():
        st.session_state.setdefault(k, v)

    if not st.session_state.token:
        sid = st.query_params.get("sid")
        if sid:
            _restore_session(sid)

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


# ── Auth ──────────────────────────────────────────────────────────────────────
def _login_page():
    st.title("🔬 Research AI")
    tab_login, tab_reg = st.tabs(["Login", "Register"])

    with tab_login:
        email    = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", type="primary", use_container_width=True):
            r = requests.post(f"{BACKEND}/auth/login",
                              json={"email": email, "password": password}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                st.session_state.token = data["access_token"]
                st.session_state.user  = email
                st.query_params["sid"] = data["session_id"]
                st.rerun()
            else:
                st.error("Invalid credentials.")

    with tab_reg:
        r_email    = st.text_input("Email",    key="reg_email")
        r_username = st.text_input("Username", key="reg_user")
        r_password = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Register", type="primary", use_container_width=True):
            r = requests.post(f"{BACKEND}/auth/register",
                              json={"email": r_email, "username": r_username,
                                    "password": r_password}, timeout=10)
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
                    "analyzed": False,
                    "llm_id":   c["llm_id"],
                }
            else:
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
            "ready": False, "analyzed": False, "llm_id": data["llm_id"],
        }
        st.session_state.current_chat_id = cid
        st.rerun()


def _delete_chat(cid: str):
    _api("delete", f"/chats/{cid}")
    st.session_state.chats.pop(cid, None)
    st.session_state.analysis_cache.pop(cid, None)
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


# ── Upload & poll ─────────────────────────────────────────────────────────────
def _upload(cid: str, file) -> bool:
    r = _api("post", f"/chats/{cid}/upload",
             files={"file": (file.name, file.getvalue(), "application/pdf")})
    return r is not None and r.status_code == 200


def _poll_ingest(cid: str) -> str:
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
            bar.progress(95, text="⏳ Still processing …")
            if i > MAX_POLLS * 3:
                bar.empty()
                return "timeout"


def _trigger_analysis(cid: str) -> bool:
    """Trigger orchestrator and poll until complete."""
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
        pct = min(int(i / 30 * 100), 95)
        bar.progress(pct, text=f"🤖 {s} … ({i * 3}s)")
        if s == "complete":
            bar.progress(100, text="✅ Analysis complete!")
            return True
        if s == "failed":
            bar.empty()
            st.error(f"Analysis failed.")
            return False
        if i > 80:   # 4 min cap
            bar.empty()
            st.warning("Analysis is taking long — check back in a moment.")
            return False


def _fetch_analysis(cid: str) -> Optional[Dict]:
    """Fetch analysis results from backend."""
    r = _api("get", f"/chats/{cid}/analysis", show_error=False)
    if r and r.status_code == 200:
        return r.json()
    return None


# ── PyVis Knowledge Graph ────────────────────────────────────────────────────
def _render_knowledge_graph(chat_id: str, papers: list):
    """Fetch triples from paper analyses and render as PyVis graph."""
    try:
        from pyvis.network import Network
        import streamlit.components.v1 as components

        # collect all triples from all papers
        all_triples = []
        for paper in papers:
            triples = paper.get("triples", [])
            all_triples.extend(triples)

        if not all_triples:
            st.info("No knowledge graph available — triples not extracted yet.")
            return

        # build PyVis network
        net = Network(
            height="500px",
            width="100%",
            bgcolor="#0e1117",
            font_color="white",
            directed=True,
        )
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)

        # color map by relation type
        relation_colors = {
            "TRAINED_ON":    "#60a5fa",
            "EVALUATED_ON":  "#34d399",
            "ACHIEVES":      "#fbbf24",
            "USES":          "#a78bfa",
            "IMPROVES_OVER": "#f87171",
            "PROPOSED_BY":   "#fb923c",
            "APPLIED_TO":    "#38bdf8",
            "COMPARED_WITH": "#e879f9",
            "BASED_ON":      "#4ade80",
            "REPLACES":      "#f43f5e",
        }

        nodes_added = set()

        for t in all_triples[:60]:   # cap at 60 triples for performance
            subject  = str(t.get("subject", "")).strip()
            relation = str(t.get("relation", "")).strip().upper()
            obj      = str(t.get("object",  "")).strip()

            if not subject or not obj:
                continue

            if subject not in nodes_added:
                net.add_node(subject, label=subject, color="#60a5fa",
                             size=20, title=subject)
                nodes_added.add(subject)

            if obj not in nodes_added:
                net.add_node(obj, label=obj, color="#4ade80",
                             size=15, title=obj)
                nodes_added.add(obj)

            edge_color = relation_colors.get(relation, "#888888")
            net.add_edge(subject, obj, label=relation,
                         color=edge_color, title=relation)

        # generate HTML
        html = net.generate_html()
        components.html(html, height=520, scrolling=False)

        st.caption(f"📊 {len(nodes_added)} nodes · {min(len(all_triples), 60)} edges · colors by relation type")

    except ImportError:
        st.warning("Install pyvis: `uv pip install pyvis`")
    except Exception as e:
        st.error(f"Graph render failed: {e}")


# ── Analysis renderers (inline, collapsible) ──────────────────────────────────
def _render_analysis(analysis: Dict):
    papers = analysis.get("papers", [])
    if not papers:
        return

    for paper in papers:
        fname    = paper.get("filename", "Paper")
        entities = paper.get("entities", {})
        summary  = paper.get("refined_summary", "")
        summaries= paper.get("summaries", {})
        gaps     = paper.get("research_gaps", [])
        dirs     = paper.get("future_directions", [])
        score    = paper.get("quality_score", 0)

        st.markdown(f"---\n### 📄 {fname}")
        col1, col2 = st.columns([4, 1])
        with col2:
            st.metric("Quality", f"{score:.1f}/10")

        # ── Summary ───────────────────────────────────────────────────────────
        with st.expander("📋 Summary", expanded=True):
            if summary:
                st.markdown(summary)
            if summaries:
                for section, text in summaries.items():
                    if text and section != "overall":
                        st.markdown(f"**{section.title()}:** {text}")

        # ── Entities ──────────────────────────────────────────────────────────
        with st.expander("🔬 Entities"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**🤖 Models**")
                for m in entities.get("models", []):
                    st.markdown(f'<span class="entity-tag">{m}</span>',
                                unsafe_allow_html=True)
            with c2:
                st.markdown("**📊 Datasets**")
                for d in entities.get("datasets", []):
                    st.markdown(f'<span class="entity-tag">{d}</span>',
                                unsafe_allow_html=True)
            with c3:
                st.markdown("**📏 Metrics**")
                for me in entities.get("metrics", []):
                    st.markdown(f'<span class="entity-tag">{me}</span>',
                                unsafe_allow_html=True)
            if entities.get("methods"):
                st.markdown("**🔧 Methods**")
                for method in entities.get("methods", []):
                    st.markdown(f'<span class="entity-tag">{method}</span>',
                                unsafe_allow_html=True)

        # ── Research Gaps ─────────────────────────────────────────────────────
        with st.expander("🔍 Research Gaps & Future Directions"):
            if gaps:
                st.markdown("**Research Gaps:**")
                for gap in gaps:
                    st.markdown(f'<div class="gap-item">🔹 {gap}</div>',
                                unsafe_allow_html=True)
            else:
                st.info("No gaps detected yet.")
            if dirs:
                st.markdown("**Future Directions:**")
                for d in dirs:
                    st.markdown(f"- {d}")

        # ── Similar Papers ────────────────────────────────────────────────────
        similar = paper.get("similar_papers", [])
        if similar:
            with st.expander("🔗 Similar Papers (arXiv)"):
                for p in similar[:5]:
                    st.markdown(
                        f"**{p.get('title', 'Unknown')}** ({p.get('year', '')})\n\n"
                        f"{p.get('abstract', '')[:200]}…\n\n"
                        f"[View on arXiv]({p.get('url', '#')})"
                    )
                    st.divider()

    # ── Knowledge Graph ──────────────────────────────────────────────────────
    with st.expander("🕸️ Knowledge Graph"):
        _render_knowledge_graph(
            chat_id=st.session_state.current_chat_id,
            papers=papers,
        )

    # ── Comparison (cross-paper) ──────────────────────────────────────────────
    comp = analysis.get("comparison", {})
    if comp:
        with st.expander("📊 Comparison"):
            if comp.get("positioning"):
                st.markdown(f"**Positioning:** {comp['positioning']}")
            if comp.get("evolution_trends"):
                st.markdown(f"**Evolution:** {comp['evolution_trends']}")
            table = comp.get("comparison_table", {})
            if table and table.get("headers") and table.get("rows"):
                import pandas as pd
                df = pd.DataFrame(table["rows"], columns=table["headers"])
                st.dataframe(df, use_container_width=True)
            if comp.get("ranking"):
                st.markdown("**Ranking:** " + " > ".join(comp["ranking"]))

    # ── Literature Review ─────────────────────────────────────────────────────
    lit = analysis.get("literature_review", {})
    if lit and lit.get("review_text"):
        with st.expander("📚 Literature Review"):
            if lit.get("themes"):
                st.markdown("**Themes:** " + " · ".join(lit["themes"]))
            st.markdown(lit["review_text"])
            if lit.get("research_gaps_summary"):
                st.markdown(f"**Gaps Summary:** {lit['research_gaps_summary']}")
            if lit.get("future_directions"):
                st.markdown(f"**Future Directions:** {lit['future_directions']}")


# ── Streaming ─────────────────────────────────────────────────────────────────
def _fix_format(text: str) -> str:
    import re
    text = re.sub(r'(\S)\s{0,2}(\d+\.\s+\*\*)', r'\1\n\n\2', text)
    text = re.sub(r'(\S)\s{0,2}(\*\*Summary)', r'\1\n\n\2', text)
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
                if event.data == "[DONE]": break
                if event.data.startswith("[ERROR]"):
                    full = event.data; break
                full += event.data.replace(chr(160), ' ')
                placeholder.markdown(full + "▌", unsafe_allow_html=True)
            full = _fix_format(full)
        placeholder.markdown(full, unsafe_allow_html=True)
        return full
    except Exception:
        r      = _api("post", f"/chats/{cid}/message",
                      json={"message": message, "llm_id": llm_id, "allow_search": allow_search})
        answer = r.json().get("answer", "Error.") if r else "Error."
        placeholder.markdown(answer, unsafe_allow_html=True)
        return answer


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.token:
    _login_page()
    st.stop()

_load_chats()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research AI")
    st.caption(f"Logged in as **{st.session_state.user}**")

    if st.button("Logout", use_container_width=True):
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

# ── PDF Upload ────────────────────────────────────────────────────────────────
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
            status = _poll_ingest(cid)
            if status == "ready":
                chat["ready"] = True
                st.success(f"✅ {f.name} indexed!")
            elif status == "failed":
                st.error(f"❌ Indexing failed for {f.name}")
            else:
                chat["ready"] = True
                st.warning("⏱ Still processing — continuing anyway.")
        st.rerun()

if not chat["ready"]:
    st.info("Upload at least one PDF to start chatting.")
    st.stop()

# ── Get Analysis button ───────────────────────────────────────────────────────
if not chat.get("analyzed"):
    st.markdown(
        '<div class="analysis-banner">✅ Paper processed! Click <b>Get Analysis</b> '
        'to extract entities, summaries, research gaps, and comparison.</div>',
        unsafe_allow_html=True,
    )
    if st.button("🔍 Get Analysis", type="primary", use_container_width=True):
        with st.spinner("Starting analysis …"):
            success = _trigger_analysis(cid)
        if success:
            chat["analyzed"] = True
            # fetch and cache analysis
            data = _fetch_analysis(cid)
            if data:
                st.session_state.analysis_cache[cid] = data
        st.rerun()

# ── Analysis results (inline collapsible) ────────────────────────────────────
if chat.get("analyzed"):
    analysis = st.session_state.analysis_cache.get(cid)
    if not analysis:
        # try fetching if not in cache (e.g. after page refresh)
        analysis = _fetch_analysis(cid)
        if analysis:
            st.session_state.analysis_cache[cid] = analysis

    if analysis:
        _render_analysis(analysis)

# ── Chat messages ─────────────────────────────────────────────────────────────
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and msg.get("source"):
            st.caption(f'📌 {msg["source"]} · 🎯 {msg.get("confidence", "")}')

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about your research paper …")

if user_input:
    if not chat["messages"]:
        chat["title"] = user_input[:40]
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