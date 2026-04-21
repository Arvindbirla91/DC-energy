"""
nav.py — Shared navigation bar with light/dark theme toggle.
Import and call render_nav() at the top of every page.
"""
import streamlit as st

# ── Theme Definitions ─────────────────────────────────────────────────
DARK = {
    "bg":         "#0d0d1a",
    "bg2":        "#111122",
    "card":       "#1a1a2e",
    "card2":      "#1e1e38",
    "border":     "rgba(255,255,255,0.07)",
    "text":       "#e0e0e0",
    "subtext":    "#888",
    "accent":     "#00CC96",
    "accent2":    "#636EFA",
    "accent3":    "#AB63FA",
    "nav_bg":     "rgba(10,10,22,0.97)",
    "nav_link":   "#999",
    "nav_hover":  "rgba(255,255,255,0.07)",
    "metric_val": "#00CC96",
    "divider":    "rgba(255,255,255,0.08)",
    "banner_bg":  "rgba(0,204,150,0.08)",
    "banner_bdr": "rgba(0,204,150,0.25)",
    "insight_bg": "rgba(99,110,250,0.07)",
    "insight_bdr":"rgba(99,110,250,0.2)",
    "toggle_icon":"☀️",
    "toggle_tip": "Switch to Light Mode",
    "plotly_tpl": "plotly_dark",
}
LIGHT = {
    "bg":         "#f4f6fb",
    "bg2":        "#eef1f8",
    "card":       "#ffffff",
    "card2":      "#f8f9fd",
    "border":     "rgba(0,0,0,0.07)",
    "text":       "#1a1a2e",
    "subtext":    "#666",
    "accent":     "#059669",
    "accent2":    "#4f46e5",
    "accent3":    "#7c3aed",
    "nav_bg":     "rgba(255,255,255,0.97)",
    "nav_link":   "#555",
    "nav_hover":  "rgba(0,0,0,0.05)",
    "metric_val": "#059669",
    "divider":    "rgba(0,0,0,0.08)",
    "banner_bg":  "rgba(5,150,105,0.06)",
    "banner_bdr": "rgba(5,150,105,0.2)",
    "insight_bg": "rgba(79,70,229,0.05)",
    "insight_bdr":"rgba(79,70,229,0.15)",
    "toggle_icon":"🌙",
    "toggle_tip": "Switch to Dark Mode",
    "plotly_tpl": "plotly_white",
}


def get_theme():
    """Return current theme dict from session_state."""
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    return DARK if st.session_state.theme == "dark" else LIGHT


def inject_theme_css(t):
    """Inject full global theme CSS."""
    st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', sans-serif !important;
        background-color: {t['bg']} !important;
        color: {t['text']} !important;
    }}
    .stApp {{ background: linear-gradient(135deg, {t['bg']} 0%, {t['bg2']} 100%) !important; }}
    header[data-testid="stHeader"] {{ display: none !important; }}
    [data-testid="collapsedControl"] {{ display: none !important; }}
    section[data-testid="stSidebar"] {{ display: none !important; }}

    /* Metrics */
    [data-testid="stMetricLabel"] {{ font-size: 0.72rem !important; color: {t['subtext']} !important; text-transform: uppercase; letter-spacing: 0.06em; }}
    [data-testid="stMetricValue"] {{ font-size: 1.6rem !important; font-weight: 700 !important; color: {t['text']} !important; }}
    [data-testid="stMetricDelta"] {{ font-size: 0.78rem !important; }}
    [data-testid="stMetric"] {{ background: {t['card']} !important; border: 1px solid {t['border']} !important; border-radius: 12px !important; padding: 1rem 1.2rem !important; }}

    /* DataFrames */
    [data-testid="stDataFrame"] {{ background: {t['card']} !important; border-radius: 10px !important; overflow: hidden; border: 1px solid {t['border']} !important; }}
    .stDataFrame thead th {{ background: {t['card2']} !important; color: {t['text']} !important; font-size: 0.8rem !important; }}

    /* Divider */
    hr {{ border-color: {t['divider']} !important; margin: 1.2rem 0 !important; }}

    /* Buttons */
    .stButton > button {{
        border-radius: 8px !important; font-weight: 600 !important;
        transition: all 0.18s ease !important;
        background: {t['card']} !important;
        color: {t['text']} !important;
        border: 1px solid {t['border']} !important;
    }}
    .stButton > button[kind="primary"] {{
        background: {t['accent']} !important;
        color: #fff !important; border: none !important;
    }}
    .stButton > button:hover {{ transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ background: {t['card']} !important; border-radius: 10px !important; padding: 0.3rem !important; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600 !important; font-size: 0.88rem !important; border-radius: 8px !important; color: {t['subtext']} !important; }}
    .stTabs [aria-selected="true"] {{ background: {t['accent']}22 !important; color: {t['accent']} !important; }}

    /* Expander */
    .stExpander {{ background: {t['card']} !important; border: 1px solid {t['border']} !important; border-radius: 10px !important; }}
    .stExpander summary {{ color: {t['text']} !important; font-weight: 600 !important; }}

    /* Selectbox, Slider */
    [data-testid="stSelectbox"] > div, [data-testid="stSlider"] {{ color: {t['text']} !important; }}

    /* Info/Success boxes */
    [data-testid="stInfo"] {{ background: {t['insight_bg']} !important; border: 1px solid {t['insight_bdr']} !important; border-radius: 8px !important; }}

    /* Page link */
    [data-testid="stPageLink"] a {{ color: {t['nav_link']} !important; text-decoration: none !important;
        padding: 0.38rem 0.85rem !important; border-radius: 7px !important;
        font-size: 0.87rem !important; font-weight: 500 !important;
        transition: all 0.15s ease !important; display: inline-block !important; }}
    [data-testid="stPageLink"] a:hover {{ background: {t['nav_hover']} !important; color: {t['text']} !important; }}

    /* NAV BAR */
    .nav-bar {{
        display: flex; align-items: center; gap: 0.2rem;
        background: {t['nav_bg']};
        border-bottom: 1px solid {t['border']};
        padding: 0.55rem 1.2rem 0.55rem 1rem;
        position: sticky; top: 0; z-index: 9999;
        backdrop-filter: blur(14px);
        margin-bottom: 0.2rem;
    }}
    .nav-brand {{
        font-size: 1rem; font-weight: 700; color: {t['accent']};
        margin-right: 1rem; white-space: nowrap; letter-spacing: -0.01em;
    }}
    .nav-spacer {{ flex: 1; }}

    /* KPI Card */
    .kpi-card {{
        background: {t['card']};
        border: 1px solid {t['border']};
        border-radius: 12px; padding: 1.1rem 1.2rem;
        text-align: center; transition: all 0.2s ease;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    .kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.1); }}
    .kpi-value {{ font-size: 1.8rem; font-weight: 700; color: {t['metric_val']}; line-height: 1.1; }}
    .kpi-label {{ font-size: 0.72rem; color: {t['subtext']}; margin-top: 0.3rem;
                  text-transform: uppercase; letter-spacing: 0.06em; }}

    /* Section title */
    .sec-title {{
        font-size: 1.2rem; font-weight: 700; color: {t['text']};
        margin: 1.8rem 0 0.8rem 0;
        display: flex; align-items: center; gap: 0.5rem;
    }}
    .sec-num {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 28px; height: 28px; border-radius: 7px;
        background: {t['accent']}22; color: {t['accent']};
        font-size: 0.85rem; font-weight: 700; flex-shrink: 0;
    }}

    /* Insight badge */
    .insight-pill {{
        display: inline-block;
        background: {t['banner_bg']};
        border: 1px solid {t['banner_bdr']};
        border-radius: 20px; padding: 0.3rem 0.9rem;
        font-size: 0.82rem; color: {t['text']}; margin: 0.2rem;
    }}

    /* Banner */
    .data-banner {{
        background: {t['banner_bg']}; border: 1px solid {t['banner_bdr']};
        border-radius: 10px; padding: 0.6rem 1rem;
        font-size: 0.88rem; color: {t['accent']}; font-weight: 500;
        display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.8rem;
    }}
</style>
""", unsafe_allow_html=True)


def render_nav():
    """
    Render sticky horizontal navigation bar with theme toggle.
    Call this as the FIRST thing after set_page_config() on every page.
    """
    t = get_theme()
    inject_theme_css(t)

    # Nav HTML wrapper
    st.markdown('<div class="nav-bar">', unsafe_allow_html=True)
    nav_cols = st.columns([1.5, 1.2, 1.2, 1.2, 4, 1.2])

    with nav_cols[0]:
        st.markdown(f'<span class="nav-brand">⚡ DC Energy</span>', unsafe_allow_html=True)
    with nav_cols[1]:
        st.page_link("pages/1_overview.py",  label="🏠 Overview")
    with nav_cols[2]:
        st.page_link("pages/2_insights.py",  label="🔍 Insights")
    with nav_cols[3]:
        st.page_link("pages/3_prediction.py", label="🔮 Prediction")
    with nav_cols[4]:
        st.markdown("")   # spacer
    with nav_cols[5]:
        icon = "☀️" if st.session_state.get("theme", "dark") == "dark" else "🌙"
        if st.button(icon, key="theme_toggle", help="Toggle Light / Dark Mode"):
            st.session_state.theme = "light" if st.session_state.get("theme", "dark") == "dark" else "dark"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
