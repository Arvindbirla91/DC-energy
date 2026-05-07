"""
Run: streamlit run app.py
"""
import streamlit as st
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

st.set_page_config(
    page_title="DC Energy Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from nav import render_nav
render_nav()

# ─Our Landing Hero ──────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 3.5rem 2rem 1.5rem 2rem;">
    <div style="font-size:3.8rem; margin-bottom:0.8rem;">⚡</div>
    <h1 style="font-size:2.6rem; font-weight:800; color:#00CC96; margin:0; line-height:1.1;">
        Data Center Energy
    </h1>
    <h2 style="font-size:1.8rem; font-weight:300; color:#aaa; margin:0.3rem 0 1.4rem 0;">
        Analysis & Prediction Platform
    </h2>
    <p style="font-size:1rem; color:#666; max-width:560px; margin:0 auto 2rem auto; line-height:1.7;">
        Explore global data center energy trends, perform in-depth analysis,
        and forecast future consumption using machine learning.
    </p>
</div>
""", unsafe_allow_html=True)

# ─ Feature Cards ─────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

cards = [
    ("🏠", "Overview",   "#00CC96", "pages/1_overview.py",
     "Professional dashboard showing global KPIs, context infographics, and dataset summary."),
    ("🔍", "Insights",   "#636EFA", "pages/2_insights.py",
     "10+ interactive charts: bar, pie, line, histogram, box, heatmap, scatter, and more."),
    ("🔮", "Prediction", "#AB63FA", "pages/3_prediction.py",
     "Two ML models: future energy forecast by country & new data center risk assessment."),
]

for col, (icon, title, color, page_path, desc) in zip([col1, col2, col3], cards):
    with col:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a2e, #1e1e38);
            border: 1px solid {color}33;
            border-top: 3px solid {color};
            border-radius: 14px;
            padding: 1.8rem 1.5rem;
            text-align: center;
            min-height: 180px;
        ">
            <div style="font-size:2.4rem; margin-bottom:0.5rem;">{icon}</div>
            <div style="font-size:1.1rem; font-weight:700; color:{color}; margin-bottom:0.5rem;">{title}</div>
            <div style="font-size:0.82rem; color:#888; line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(page_path, label=f"Open {title} →")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#444; font-size:0.85rem;">
    Use the <b style="color:#00CC96">navigation bar above</b> to switch between sections
</div>
""", unsafe_allow_html=True)
