"""auth_ui.py — Login and Register pages"""
from __future__ import annotations
import requests
import streamlit as st


def render_login_page(backend: str) -> None:
    st.markdown("""
    <div style='text-align:center;padding:60px 0 40px'>
        <div style='font-family:Space Mono,monospace;font-size:2.2rem;
                    font-weight:700;letter-spacing:-0.04em;
                    background:linear-gradient(135deg,#6366f1,#a5b4fc);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
            Research AI
        </div>
        <div style='color:#8888aa;font-size:0.9rem;margin-top:8px'>
            Multi-Agent Scientific Paper Intelligence System
        </div>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        tab_login, tab_reg = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            email    = st.text_input("Email", placeholder="you@example.com", key="login_email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("Sign In", type="primary", use_container_width=True, key="btn_login"):
                r = requests.post(
                    f"{backend}/auth/login",
                    json={"email": email, "password": password},
                    timeout=10,
                )
                if r.status_code == 200:
                    data = r.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.user  = email
                    st.query_params["sid"] = data["session_id"]
                    st.rerun()
                else:
                    st.error("Invalid credentials — please try again.")

        with tab_reg:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            r_email    = st.text_input("Email",    placeholder="you@example.com", key="reg_email")
            r_username = st.text_input("Username", placeholder="researcher_42",   key="reg_user")
            r_password = st.text_input("Password", type="password",
                                       placeholder="min 8 characters", key="reg_pass")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("Create Account", type="primary", use_container_width=True, key="btn_reg"):
                r = requests.post(
                    f"{backend}/auth/register",
                    json={"email": r_email, "username": r_username, "password": r_password},
                    timeout=10,
                )
                if r.status_code == 201:
                    st.success("✅ Account created — sign in to continue.")
                else:
                    st.error(r.json().get("detail", "Registration failed."))