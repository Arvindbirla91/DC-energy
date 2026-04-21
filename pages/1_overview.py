"""
pages/1_overview.py — Overview Dashboard
Dataset top-10 rows · KPI cards · Key insights · Minimized infographics
"""
import streamlit as st
import os, sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing import load_and_clean_data, get_data_profile
from nav import render_nav, get_theme

st.set_page_config(page_title="Overview | DC Energy", page_icon="⚡",
                   layout="wide", initial_sidebar_state="collapsed")

render_nav()
t = get_theme()

# ── Load Data ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    return load_and_clean_data()

with st.spinner("Loading..."):
    df = get_data()
    profile = get_data_profile(df)

# ── Page Header ────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:1.6rem 0 0.4rem 0;">
  <h1 style="font-size:2rem;font-weight:800;color:{t['text']};margin:0;">
    ⚡ Data Center Energy Analysis
  </h1>
  <p style="color:{t['subtext']};font-size:0.95rem;margin-top:0.3rem;">
    Comprehensive analytics platform for global data center energy consumption across
    <b style="color:{t['accent']}">{profile['total_countries']} countries</b>
  </p>
</div>
""", unsafe_allow_html=True)


# ── Dataset loaded banner ──────────────────────────────────────────────
st.markdown(f"""
<div class="data-banner">
  ✅ Dataset loaded — <b>{len(df)} rows × {len(df.columns)} columns</b>
  &nbsp;·&nbsp; Source: Global Data Center Intelligence Report
</div>
""", unsafe_allow_html=True)

# ── TOP 10 ROWS — ALL columns ───────────────────────────────────────────
with st.expander("📋 Dataset Preview — Top 10 Rows (All Columns)", expanded=True):
    # Show every column, smart-format numerics
    fmt = {
        col: '{:.2f}' for col in df.select_dtypes(include='float').columns
    }
    fmt['power_capacity_MW']      = '{:.1f}'
    fmt['avg_renewable_energy_pct'] = '{:.1f}'
    fmt['growth_rate_pct']        = '{:.2f}'
    fmt['avg_latency_ms']         = '{:.0f}'
    fmt['internet_penetration_pct'] = '{:.3f}'
    fmt = {k: v for k, v in fmt.items() if k in df.columns}
    st.dataframe(
        df.head(10).style.format(fmt),
        use_container_width=True, height=320,
    )
    st.caption(f"📌 {len(df.columns)} columns shown · {len(df)} rows total")

st.divider()

# ── KPI Cards ──────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-title">📊 Key Metrics at a Glance</div>', unsafe_allow_html=True)
cols = st.columns(5)
kpis = [
    ("🌍", profile["total_countries"], "Countries"),
    ("🏗️", f"{profile['max_data_centers']:,}", f"Max DCs · {profile['top_country']}"),
    ("⚡", f"{profile['avg_power_capacity_mw']:.0f} MW", "Avg Power Capacity"),
    ("🌿", f"{profile['avg_renewable_pct']:.1f}%", "Avg Renewable Energy"),
    ("📈", f"{profile['avg_growth_rate']:.1f}%/yr", "Avg DC Growth Rate"),
]
for col, (icon, val, label) in zip(cols, kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div style="font-size:1.6rem;margin-bottom:0.3rem">{icon}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Key Insight Lines ──────────────────────────────────────────────────
st.markdown(f'<div class="sec-title">💡 Key Findings at a Glance</div>', unsafe_allow_html=True)

top_renewable_country = df.nlargest(1, 'avg_renewable_energy_pct').iloc[0]
top_growth_country    = df.nlargest(1, 'growth_rate_pct').iloc[0]
top_dc_country        = df.nlargest(1, 'total_data_centers').iloc[0]
# avg_renewable_energy_pct is already 0-100 scale
arid_avg  = df[df['Climate_Type']=='Arid']['avg_renewable_energy_pct'].mean()
temp_avg  = df[df['Climate_Type']=='Temperate']['avg_renewable_energy_pct'].mean()

insights = [
    f"🥇 <b>{top_dc_country['country']}</b> leads globally with <b>{int(top_dc_country['total_data_centers']):,}</b> data centers",
    f"🌿 <b>{top_renewable_country['country']}</b> has the highest renewable energy at <b>{top_renewable_country['avg_renewable_energy_pct']:.0f}%</b>",
    f"📈 <b>{top_growth_country['country']}</b> has the fastest DC growth at <b>{top_growth_country['growth_rate_pct']:.1f}%/yr</b>",
    f"🌡️ Temperate-climate countries avg <b>{temp_avg:.1f}%</b> renewable vs Arid at <b>{arid_avg:.1f}%</b>",
    f"⚡ Global average power capacity: <b>{df['power_capacity_MW'].mean():.1f} MW</b> per country",
    f"🌐 <b>{df['Climate_Type'].value_counts().index[0]}</b> is the most common climate type across <b>{df['Climate_Type'].value_counts().iloc[0]}</b> countries",
]

icols = st.columns(2)
for i, insight in enumerate(insights):
    with icols[i % 2]:
        st.markdown(f'<div class="insight-pill">→ {insight}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Industry Context — 2-column split (fixed small size) ─────────────
st.markdown(f'<div class="sec-title">🌐 Industry Context</div>', unsafe_allow_html=True)

GRAPHS_DIR = "graphs"
img_left, img_right = st.columns(2, gap="large")

with img_left:
    forecast_path = os.path.join(GRAPHS_DIR, 'energy_forecast_2030.png')
    if os.path.exists(forecast_path):
        # Fixed width so image never stretches full column
        st.image(forecast_path, width=380)
    st.markdown(f"""
    <div style="padding:0.4rem 0 0 0;">
      <b style="color:{t['accent']};font-size:0.85rem;">📈 Global Energy Forecast (IEA / Enerdata)</b>
      <ul style="color:{t['subtext']};font-size:0.81rem;margin:0.3rem 0 0 0;line-height:1.6;padding-left:1rem;">
        <li>Data center demand is projected to <b style="color:{t['text']}">nearly double</b> by 2030.</li>
        <li>AI &amp; cloud workloads are the primary accelerants — a ChatGPT query uses ~10× more energy than Google.</li>
        <li>Data Centers have the steepest growth slope across the entire ICT sector.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

with img_right:
    compare_path = os.path.join(GRAPHS_DIR, 'datacenter_electricity_comparison.png')
    if os.path.exists(compare_path):
        st.image(compare_path, width=380)
    st.markdown(f"""
    <div style="padding:0.4rem 0 0 0;">
      <b style="color:{t['accent']};font-size:0.85rem;">⚡ Data Centers vs. Entire Countries (2020 TWh)</b>
      <ul style="color:{t['subtext']};font-size:0.81rem;margin:0.3rem 0 0 0;line-height:1.6;padding-left:1rem;">
        <li>Global data centers consume <b style="color:{t['text']}">200–250 TWh/yr</b> — more than South Africa's national grid.</li>
        <li>Concentrated in <b style="color:{t['text']}">fewer than 20 countries</b>, creating geographic energy inequality.</li>
        <li>Transitioning to renewables is the defining sustainability challenge of the digital economy.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

# Global data growth — small fixed size, centered
growth_path = os.path.join(GRAPHS_DIR, 'global_data_growth.png')
if os.path.exists(growth_path):
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📡 Global Data Growth Context (2010–2026)", expanded=False):
        _, mid, _ = st.columns([2, 3, 2])
        with mid:
            st.image(growth_path, width=320)
        st.caption("Data volume — driven by IoT, streaming, and AI — is surging 100× since 2010, directly fueling demand for more data centers.")


st.divider()

# ── Dataset Summary Stats (collapsible) ───────────────────────────────
with st.expander("📊 Summary Statistics", expanded=False):
    numeric_df = df.select_dtypes(include='number')
    st.dataframe(numeric_df.describe().T.round(3), use_container_width=True)

with st.expander("🗂️ Column Data Types", expanded=False):
    dtypes_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str).values,
        'Non-Null': df.notna().sum().values,
        'Null': df.isna().sum().values,
    })
    st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

with st.expander("📋 About This Project", expanded=False):
    st.markdown(f"""
    This platform analyzes **{len(df)} countries** across **{len(df.columns)} features** of data center energy infrastructure.

    | Component | Details |
    |---|---|
    | **Data** | Raw CSV → Cleaned · Imputed · Outlier-handled |
    | **EDA** | 10+ interactive charts across climate, power, renewables |
    | **Models** | Linear · Random Forest · Gradient Boosting Regression |
    | **Predictions** | Country-level forecast + New DC risk assessment |

    **Key variables:** `total_data_centers` · `power_capacity_MW` · `avg_renewable_energy_pct` · `growth_rate_pct` · `Climate_Type`
    """)

st.divider()

# ── Continue to Insights CTA ──────────────────────────────────────────
cta_col1, cta_col2, cta_col3 = st.columns([1, 2, 1])
with cta_col2:
    st.markdown(f"""
    <div style="background:{t['banner_bg']};border:1px solid {t['banner_bdr']};
    border-radius:14px;padding:1.4rem 2rem;text-align:center;">
      <div style="font-size:1.3rem;margin-bottom:0.4rem;">🔍</div>
      <div style="font-weight:700;color:{t['text']};margin-bottom:0.3rem;">Ready to explore the data?</div>
      <div style="font-size:0.85rem;color:{t['subtext']};">
        Head to <b>Insights</b> for interactive charts and deep-dive EDA
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.page_link("pages/2_insights.py", label="▶ Continue to Insights →", icon="🔍")
