"""
pages/2_insights.py — Insights Dashboard
Numbered interactive chart sections, Filter & Explore, analytics mind charts.
"""
import streamlit as st
import os, sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing import load_and_clean_data
from preprocessing import engineer_features
import visualization as viz
from nav import render_nav, get_theme

st.set_page_config(page_title="Insights | DC Energy", page_icon="🔍",
                   layout="wide", initial_sidebar_state="collapsed")

render_nav()
t = get_theme()

# ── Load Data ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    df = load_and_clean_data()
    df = engineer_features(df)
    return df

with st.spinner("Loading insights..."):
    df = get_data()

# ── Page Header ────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:1.6rem 0 0.5rem 0;">
  <h1 style="font-size:2rem;font-weight:800;color:{t['text']};margin:0;">
    🔍 Data Insights & Exploration
  </h1>
  <p style="color:{t['subtext']};font-size:0.92rem;margin-top:0.3rem;">
    Interactive analytics across {len(df)} countries · Adjust filters to explore the data
  </p>
</div>
""", unsafe_allow_html=True)

# ── Filter & Explore ───────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{t['card']};border:1px solid {t['border']};
border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem;">
  <b style="color:{t['text']}">🎛️ Filter & Explore</b>
</div>""", unsafe_allow_html=True)

f_col1, f_col2, f_col3 = st.columns([2, 1.5, 1.5])
with f_col1:
    all_climates = ['All'] + sorted(df['Climate_Type'].unique().tolist())
    sel_climate = st.multiselect("Filter by Climate Type", all_climates[1:],
                                  default=[], placeholder="All climates")
with f_col2:
    top_n = st.slider("Top N Countries to show", 5, 25, 10)
with f_col3:
    min_dc = st.slider("Min. Data Centers", 0, int(df['total_data_centers'].max()), 0)

# Apply filters
fdf = df.copy()
if sel_climate:
    fdf = fdf[fdf['Climate_Type'].isin(sel_climate)]
if min_dc > 0:
    fdf = fdf[fdf['total_data_centers'] >= min_dc]

if len(fdf) == 0:
    st.warning("No data matches the current filters. Showing all data.")
    fdf = df.copy()

st.markdown(f"""
<div class="data-banner" style="margin-top:0.5rem">
  📋 Showing <b>{len(fdf)} countries</b> with current filters
</div>""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════
#  SECTION 1 & 2  — Data Center Rankings
# ══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="font-size:1.3rem;font-weight:700;color:{t['text']};margin:0.5rem 0 1rem 0;">
  📊 Data Visualizations
</div>""", unsafe_allow_html=True)

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown(f'<div class="sec-title"><span class="sec-num">1</span> Top Countries by Data Centers</div>',
                unsafe_allow_html=True)
    st.plotly_chart(viz.top_countries_by_datacenters(fdf, t, n=top_n),
                    use_container_width=True, key="ch1")

with row1_col2:
    st.markdown(f'<div class="sec-title"><span class="sec-num">2</span> Top Countries by Power Capacity</div>',
                unsafe_allow_html=True)
    st.plotly_chart(viz.top_countries_by_power(fdf, t, n=top_n),
                    use_container_width=True, key="ch2")

# ══════════════════════════════════════════════════════════════════════
#  SECTION 3 & 4  — Growth & Renewables
# ══════════════════════════════════════════════════════════════════════
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.markdown(f'<div class="sec-title"><span class="sec-num">3</span> Fastest Growing DC Markets</div>',
                unsafe_allow_html=True)
    st.plotly_chart(viz.top_growth_rate_bar(fdf, t, n=top_n),
                    use_container_width=True, key="ch3")

with row2_col2:
    st.markdown(f'<div class="sec-title"><span class="sec-num">4</span> Renewable Energy Leaders</div>',
                unsafe_allow_html=True)
    st.plotly_chart(viz.renewable_leaders_bar(fdf, t, n=top_n),
                    use_container_width=True, key="ch4")

# ══════════════════════════════════════════════════════════════════════
#  SECTION 5 & 6  — Climate & Composition
# ══════════════════════════════════════════════════════════════════════
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.markdown(f'<div class="sec-title"><span class="sec-num">5</span> Climate Type Distribution</div>',
                unsafe_allow_html=True)
    st.plotly_chart(viz.climate_distribution_donut(fdf, t),
                    use_container_width=True, key="ch5")

with row3_col2:
    st.markdown(f'<div class="sec-title"><span class="sec-num">6</span> Hyperscale vs Colocation DCs</div>',
                unsafe_allow_html=True)
    st.plotly_chart(viz.hyperscale_vs_colocation(fdf, t, n=top_n),
                    use_container_width=True, key="ch6")

# ══════════════════════════════════════════════════════════════════════
#  SECTION 7  — Renewable vs Non-Renewable Stacked
# ══════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="sec-title"><span class="sec-num">7</span> Renewable vs Non-Renewable Power Capacity</div>',
            unsafe_allow_html=True)
st.plotly_chart(viz.renewable_stacked_bar(fdf, t, n=top_n),
                use_container_width=True, key="ch7")

# ══════════════════════════════════════════════════════════════════════
#  SECTION 8 & 9  — Scatter Plots
# ══════════════════════════════════════════════════════════════════════
row4_col1, row4_col2 = st.columns(2)

with row4_col1:
    st.markdown(f'<div class="sec-title"><span class="sec-num">8</span> Internet Penetration vs Data Centers</div>',
                unsafe_allow_html=True)
    st.plotly_chart(viz.internet_vs_datacenters_scatter(fdf, t),
                    use_container_width=True, key="ch8")

with row4_col2:
    st.markdown(f'<div class="sec-title"><span class="sec-num">9</span> Latency vs Data Centers</div>',
                unsafe_allow_html=True)
    st.plotly_chart(viz.latency_vs_datacenters(fdf, t),
                    use_container_width=True, key="ch9")

# ══════════════════════════════════════════════════════════════════════
#  SECTION 10  — Renewable Energy Distribution (kept, 11 removed)
# ══════════════════════════════════════════════════════════════════════
row5_col1, row5_col2 = st.columns(2)

with row5_col1:
    st.markdown(f'<div class="sec-title"><span class="sec-num">10</span> Renewable Energy Share Distribution</div>',
                unsafe_allow_html=True)
    st.markdown(f"<p style='color:{t['subtext']};font-size:0.82rem;margin-bottom:0.4rem;'>"
                "How evenly is renewable energy distributed across countries? Most countries cluster below 50%.</p>",
                unsafe_allow_html=True)
    st.plotly_chart(viz.renewable_histogram(fdf, t),
                    use_container_width=True, key="ch10")

with row5_col2:
    st.markdown(f'<div class="sec-title"><span class="sec-num">11</span> Fiber Connections Ranking</div>',
                unsafe_allow_html=True)
    st.markdown(f"<p style='color:{t['subtext']};font-size:0.82rem;margin-bottom:0.4rem;'>"
                "Countries with the most fiber connections tend to have larger and more efficient data center ecosystems.</p>",
                unsafe_allow_html=True)
    st.plotly_chart(viz.fiber_connections_bar(fdf, t, n=top_n),
                    use_container_width=True, key="ch11")

# ══════════════════════════════════════════════════════════════════════
#  SECTION 12  — Growth Rate: Manager-Friendly Summary Bar Chart
# ══════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="sec-title"><span class="sec-num">12</span> DC Market Growth — Which Countries Are Worth Watching?</div>',
            unsafe_allow_html=True)
st.markdown(f"""
<p style='color:{t['subtext']};font-size:0.88rem;margin-bottom:0.6rem;'>
  Annual data center growth rate (%/yr) by country — ranked highest to lowest.
  Countries above the <b style='color:{t["text"]}'>industry average</b> line
  are high-priority expansion markets. Use this to prioritize investment, partnerships,
  or regulatory attention.
</p>""", unsafe_allow_html=True)

import plotly.graph_objects as go

# Manager-friendly: show all countries sorted, color above/below avg, add avg line
_growth_df = fdf[['country','growth_rate_pct']].dropna().sort_values('growth_rate_pct', ascending=False)
_avg = _growth_df['growth_rate_pct'].mean()
_colors = ['#00CC96' if v >= _avg else '#636EFA' for v in _growth_df['growth_rate_pct']]

_fig12 = go.Figure(go.Bar(
    x=_growth_df['country'],
    y=_growth_df['growth_rate_pct'],
    marker=dict(color=_colors, line=dict(width=0)),
    text=_growth_df['growth_rate_pct'].apply(lambda x: f'{x:.1f}%'),
    textposition='outside',
    textfont=dict(color=t['text'], size=9),
    hovertemplate='<b>%{x}</b><br>Growth: %{y:.2f}%/yr<extra></extra>',
))
_fig12.add_hline(y=_avg, line_dash='dash', line_color='#FFA15A', line_width=2,
                 annotation_text=f'Avg: {_avg:.1f}%/yr',
                 annotation_font_color='#FFA15A', annotation_font_size=12)
_layout12 = dict(
    title=dict(text=f'📊 All Countries — Annual DC Growth Rate (%/yr) · Green = above average',
               font=dict(color=t['text'], size=13)),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=t['text']), height=420, margin=dict(l=5,r=5,t=50,b=80),
    xaxis=dict(tickangle=-40, tickfont=dict(color=t['subtext'], size=9),
               gridcolor=t['divider'], showgrid=False),
    yaxis=dict(title='Growth Rate (%/yr)', gridcolor=t['divider'],
               tickfont=dict(color=t['subtext'])),
)
_fig12.update_layout(**_layout12)
st.plotly_chart(_fig12, use_container_width=True, key="ch12")

# ══════════════════════════════════════════════════════════════════════
#  SECTION 13  — Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="sec-title"><span class="sec-num">13</span> Feature Correlation Heatmap</div>',
            unsafe_allow_html=True)
st.markdown(f"<p style='color:{t['subtext']};font-size:0.84rem;margin-bottom:0.5rem;'>"
            "Understand which features are strongly correlated — key for feature selection in ML models. "
            "Dark red = strong positive, dark blue = strong negative correlation.</p>",
            unsafe_allow_html=True)
st.plotly_chart(viz.plotly_correlation_heatmap(fdf, t), use_container_width=True, key="ch13")

# ══════════════════════════════════════════════════════════════════════
#  Quick Stats Summary
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(f'<div class="sec-title">📌 Quick Statistics Summary</div>', unsafe_allow_html=True)

qcols = st.columns(4)
quick_stats = [
    ("🌍 Countries", len(fdf)),
    ("⚡ Max Power", f"{fdf['power_capacity_MW'].max():.0f} MW"),
    # avg_renewable_energy_pct stored as 0-100, no ×100 needed
    ("🌿 Max Renewable", f"{fdf['avg_renewable_energy_pct'].max():.1f}%"),
    # growth_rate_pct stored as raw % (e.g. 5.2), no ×100 needed
    ("📈 Max Growth", f"{fdf['growth_rate_pct'].max():.1f}%/yr"),
]
for col, (label, val) in zip(qcols, quick_stats):
    with col:
        st.metric(label, val)

st.divider()
st.page_link("pages/3_prediction.py", label="▶ Continue to Predictions →", icon="🔮")
