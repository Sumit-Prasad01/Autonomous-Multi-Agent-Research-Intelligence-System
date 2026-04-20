"""styles.py — Global CSS + HTML templates"""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

:root {
    --bg-primary:    #0a0a0f;
    --bg-secondary:  #111118;
    --bg-card:       #16161f;
    --bg-hover:      #1e1e2a;
    --accent:        #6366f1;
    --accent-dim:    #4f52c4;
    --accent-glow:   rgba(99,102,241,0.15);
    --success:       #10b981;
    --warning:       #f59e0b;
    --danger:        #ef4444;
    --text-primary:  #f0f0f8;
    --text-secondary:#8888aa;
    --text-muted:    #55556a;
    --border:        rgba(255,255,255,0.06);
    --border-accent: rgba(99,102,241,0.3);
    --radius:        10px;
    --radius-lg:     16px;
}

/* ── Base ─────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}

/* ── Headings ─────────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.02em;
}

/* ── Metrics ──────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 18px !important;
}
[data-testid="metric-container"] label {
    color: var(--text-secondary) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.5rem !important;
    color: var(--text-primary) !important;
}

/* ── Buttons ──────────────────────────────────────────────────────── */
.stButton > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    background: var(--bg-hover) !important;
    border-color: var(--border-accent) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px var(--accent-glow) !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--accent-dim) !important;
    border-color: var(--accent-dim) !important;
}

/* ── Expanders ────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    margin-bottom: 10px !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 14px 18px !important;
    color: var(--text-primary) !important;
}
[data-testid="stExpander"] summary:hover {
    background: var(--bg-hover) !important;
}

/* ── Chat messages ────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    margin-bottom: 8px !important;
    padding: 4px 8px !important;
}

/* ── Input ────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: var(--radius-lg) !important;
}
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Dataframe ────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Custom components ────────────────────────────────────────────── */
.entity-tag {
    display: inline-block;
    background: rgba(99,102,241,0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.25);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 2px 3px;
    font-family: 'DM Sans', sans-serif;
}

.gap-item {
    border-left: 3px solid var(--accent);
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 0 var(--radius) var(--radius) 0;
    background: rgba(99,102,241,0.06);
    font-size: 0.88rem;
    line-height: 1.6;
}

.paper-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 20px 0 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
}

.paper-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}

.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 20px 0;
}

.analysis-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(99,102,241,0.05));
    border: 1px solid var(--border-accent);
    border-radius: var(--radius-lg);
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.9rem;
}

.badge-success { color: var(--success); }
.badge-warning { color: var(--warning); }
.badge-danger  { color: var(--danger);  }

/* ── Thinking animation ───────────────────────────────────────────── */
@keyframes thinking-dot {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40%           { transform: scale(1.0); opacity: 1.0; }
}
.thinking-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    color: var(--text-secondary);
    font-size: 0.9rem;
    font-style: italic;
}
.thinking-dots { display: flex; gap: 5px; }
.thinking-dots span {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
    animation: thinking-dot 1.2s infinite ease-in-out;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

/* ── Scrollbar ────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Progress bar ─────────────────────────────────────────────────── */
.stProgress > div > div {
    background: var(--accent) !important;
    border-radius: 4px;
}

/* ── Alerts ───────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}
</style>
"""

THINKING_HTML = """
<div class="thinking-wrap">
    <span>Thinking</span>
    <div class="thinking-dots"><span></span><span></span><span></span></div>
</div>
"""