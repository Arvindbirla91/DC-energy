"""
pages/3_prediction.py
---------------------
Prediction page with two independent ML model interfaces:
  Tab 1 — Model 1: Future Energy Consumption Forecast (by country)
  Tab 2 — Model 2: New Data Center Energy Prediction (with risk)
"""

import streamlit as st
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing import load_and_clean_data
from preprocessing import engineer_features, build_country_climate_map
from model_training import train_models
from prediction_models import (
    load_best_model,
    load_model_metadata,
    load_feature_names,
    forecast_future_energy,
    predict_new_datacenter,
    risk_gauge_chart,
    energy_breakdown_chart,
)
from nav import render_nav, get_theme

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(page_title="Prediction | DC Energy", page_icon="🔮",
                   layout="wide", initial_sidebar_state="collapsed")

render_nav()
t = get_theme()

# ── Prediction-specific CSS ─────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .section-title {{
        font-size: 1.2rem; font-weight: 700; color: {t['text']};
        border-left: 4px solid {t['accent3']}; padding-left: 0.8rem;
        margin: 1.5rem 0 0.8rem 0;
    }}
    .result-card {{
        background: {t['card']};
        border: 1px solid {t['accent3']}44;
        border-radius: 14px; padding: 1.5rem; margin: 0.5rem 0;
    }}
    .risk-high   {{ color: #EF553B; font-weight: 700; font-size: 1.4rem; }}
    .risk-medium {{ color: #FFA15A; font-weight: 700; font-size: 1.4rem; }}
    .risk-low    {{ color: #00CC96; font-weight: 700; font-size: 1.4rem; }}
    .pred-value  {{ font-size: 2.4rem; font-weight: 700; color: {t['accent3']}; }}
    .pred-unit   {{ font-size: 0.9rem; color: {t['subtext']}; }}
    .info-row    {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.8rem; }}
    .info-pill   {{
        background: {t['card2']};
        border-radius: 20px; padding: 0.3rem 0.9rem;
        font-size: 0.82rem; color: {t['text']};
    }}
    .model-badge {{
        display: inline-block;
        margin-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)


# ── Load Data & Model ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    df = load_and_clean_data()
    df = engineer_features(df)
    return df


@st.cache_resource(show_spinner=False)
def get_model():
    model = load_best_model()
    feature_names = load_feature_names()
    return model, feature_names


with st.spinner("Preparing prediction engine..."):
    df = get_data()
    model, feature_names = get_model()

country_climate_map = build_country_climate_map(df)
all_countries = sorted(df['country'].unique().tolist())
metadata = load_model_metadata()


# ── Page Header ─────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 0.5rem 0;">
    <h1 style="font-size:2rem; font-weight:700; color:#AB63FA; margin:0;">
        🔮 Energy Consumption Prediction
    </h1>
    <p style="color:#aaa; font-size:0.95rem;">
        Machine learning powered forecasting for data center energy analysis
    </p>
</div>
""", unsafe_allow_html=True)

# ── Model Status Banner ───────────────────────────────────────────────
if model is None:
    st.warning(
        "⚠️ No trained model found. Click **Train Model** below to train on the dataset first.",
        icon="⚠️"
    )
    if st.button("🚀 Train Model Now", type="primary"):
        with st.spinner("Training models... This may take a minute."):
            try:
                trained_model, results_df, feat_names = train_models()
                st.success("✅ Model trained and saved successfully! Please refresh the page.")
                st.balloons()
            except Exception as e:
                st.error(f"Training failed: {e}")
    st.stop()
else:
    if metadata:
        best_name = metadata.get("best_model", "Unknown")
        metrics = metadata.get("metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Model", best_name)
        col2.metric("MAE", f"{metrics.get('MAE', '-')}")
        col3.metric("RMSE", f"{metrics.get('RMSE', '-')}")
        col4.metric("R²", f"{metrics.get('R2', '-')}")

    with st.expander("🔄 Retrain Model", expanded=False):
        if st.button("Retrain with current dataset", type="secondary"):
            with st.spinner("Retraining..."):
                try:
                    train_models()
                    st.cache_resource.clear()
                    st.success("✅ Retrained! Refresh to load new model.")
                except Exception as e:
                    st.error(f"Retraining failed: {e}")

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Model 1 — Future Forecast", "🏗️ Model 2 — New Data Center"])


# ═══════════════════════════════════════════════
# TAB 1 — FUTURE ENERGY CONSUMPTION FORECAST
# ═══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">🔮 Future Energy Consumption Forecast</div>',
                unsafe_allow_html=True)
    st.markdown("""
    Select a country to forecast its projected data center energy consumption
    over the next several years based on historical growth patterns.
    """)

    col_in, col_info = st.columns([1, 2])
    with col_in:
        st.markdown("**⚙️ Forecast Settings**")
        selected_country_1 = st.selectbox(
            "Select Country", all_countries, index=0, key="m1_country"
        )
        forecast_years = st.slider("Forecast Horizon (Years)", 3, 15, 10, key="m1_years")

        # Show current stats for selected country
        country_data = df[df['country'] == selected_country_1]
        if len(country_data) > 0:
            r = country_data.iloc[0]
            st.markdown("**📌 Current Statistics**")
            raw_growth = float(r['growth_rate_pct'])
            growth_display = raw_growth if raw_growth >= 1.0 else raw_growth * 100
            st.markdown(f"""
            - 🏗️ Total DCs: **{int(r['total_data_centers']):,}**
            - ⚡ Power Capacity: **{r['power_capacity_MW']:.1f} MW**
            - 🌿 Renewable: **{r['avg_renewable_energy_pct']:.1f}%**
            - 📈 Growth Rate: **{growth_display:.2f}%/yr**
            - 🌦️ Climate: **{r['Climate_Type']}**
            """)

        run_forecast = st.button("🚀 Generate Forecast", type="primary", key="m1_run")

    with col_info:
        st.markdown("**ℹ️ How This Model Works**")
        st.info("""
        **Model 1** uses a compound-growth formula with AI-demand acceleration
        to produce a realistic exponential forecast curve.

        **Formula:**  `Energy(year) = Base_MW × Π(1 + rate_t)`

        **Key factors:**
        - Country base growth rate (blended with IEA sector benchmarks)
        - AI workload acceleration: +0.7% additional demand per year
        - Climate-adjusted sector CAGR (Arid = highest, Cold = lowest)
        - ±12% confidence band around central projection
        """)

    if run_forecast:
        with st.spinner(f"Forecasting energy for {selected_country_1}..."):
            try:
                forecast_df, forecast_fig = forecast_future_energy(
                    df, selected_country_1, model, feature_names, forecast_years
                )
                if forecast_df is not None:
                    st.markdown('<div class="section-title">📈 Forecast Results</div>',
                                unsafe_allow_html=True)
                    st.plotly_chart(forecast_fig, use_container_width=True)

                    c1, c2 = st.columns([1.5, 1])
                    with c1:
                        st.markdown("**📊 Year-wise Projections**")
                        st.dataframe(forecast_df, use_container_width=True, height=350)
                    with c2:
                        col_name = 'Projected Power Demand (MW)'
                        final_year = forecast_df.iloc[-1]
                        start_year = forecast_df.iloc[0]
                        growth     = final_year[col_name] - start_year[col_name]
                        growth_pct = (growth / start_year[col_name]) * 100

                        st.markdown("**📌 Forecast Summary**")
                        st.metric(
                            f"📅 Base Year ({int(start_year['Year'])})",
                            f"{start_year[col_name]:,.0f} MW",
                        )
                        st.metric(
                            f"🔮 Projected ({int(final_year['Year'])})",
                            f"{final_year[col_name]:,.0f} MW",
                            delta=f"+{growth_pct:.1f}% over {forecast_years} yrs"
                        )
                        st.metric(
                            f"🏗️ Est. Data Centers ({int(final_year['Year'])})",
                            f"{int(final_year['Total Data Centers (est.)']):,}",
                            delta=f"+{int(final_year['Total Data Centers (est.)'] - start_year['Total Data Centers (est.)']):,}"
                        )
                        st.metric(
                            "📈 Final Year Growth Rate",
                            f"{final_year['Effective Growth Rate (%/yr)']:.1f}%/yr"
                        )
                else:
                    st.error(f"Could not find data for {selected_country_1}.")
            except Exception as e:
                st.error(f"Forecast failed: {e}")


# ═══════════════════════════════════════════════
# TAB 2 — NEW DATA CENTER PREDICTION
# ═══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🏗️ New Data Center Energy Prediction</div>',
                unsafe_allow_html=True)
    st.markdown("""
    Plan a new data center — enter the key parameters below and get an
    **energy consumption prediction** plus a **risk assessment**.
    """)

    col_inputs, col_info2 = st.columns([1, 1.6])
    with col_inputs:
        st.markdown("**⚙️ Data Center Parameters**")

        selected_country_2 = st.selectbox(
            "Country / Location", all_countries, index=0, key="m2_country"
        )

        # Auto-show detected climate
        auto_climate = country_climate_map.get(selected_country_2, "Temperate")
        st.info(f"🌦️ Auto-detected Climate: **{auto_climate}**")

        power_mw = st.slider(
            "Power Capacity (MW)", min_value=1.0, max_value=500.0,
            value=50.0, step=1.0, key="m2_power"
        )
        renewable_pct_input = st.slider(
            "Renewable Energy (%)", min_value=0, max_value=100,
            value=40, step=1, key="m2_renewable"
        )
        renewable_ratio = renewable_pct_input / 100.0

        run_predict = st.button("⚡ Predict Energy Consumption", type="primary", key="m2_run")

    with col_info2:
        st.markdown("**ℹ️ Risk Level Guide**")
        risk_cols = st.columns(3)
        with risk_cols[0]:
            st.markdown("""
            <div style="background:rgba(0,204,150,0.1); border:1px solid #00CC96; 
            border-radius:10px; padding:0.8rem; text-align:center;">
            <div style="color:#00CC96; font-weight:700; font-size:1.1rem;">🟢 LOW</div>
            <div style="color:#aaa; font-size:0.8rem; margin-top:0.4rem;">
            Renewable ≥ 60%<br>Power &lt; 50 MW<br>Efficient climate
            </div></div>
            """, unsafe_allow_html=True)
        with risk_cols[1]:
            st.markdown("""
            <div style="background:rgba(255,161,90,0.1); border:1px solid #FFA15A; 
            border-radius:10px; padding:0.8rem; text-align:center;">
            <div style="color:#FFA15A; font-weight:700; font-size:1.1rem;">🟡 MEDIUM</div>
            <div style="color:#aaa; font-size:0.8rem; margin-top:0.4rem;">
            Renewable 30–59%<br>Power 50–150 MW<br>Moderate conditions
            </div></div>
            """, unsafe_allow_html=True)
        with risk_cols[2]:
            st.markdown("""
            <div style="background:rgba(239,85,59,0.1); border:1px solid #EF553B; 
            border-radius:10px; padding:0.8rem; text-align:center;">
            <div style="color:#EF553B; font-weight:700; font-size:1.1rem;">🔴 HIGH</div>
            <div style="color:#aaa; font-size:0.8rem; margin-top:0.4rem;">
            Renewable &lt; 30%<br>Power &gt; 150 MW<br>Hot/Arid climate
            </div></div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📌 Real-time Parameter Preview**")
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Country | {selected_country_2} |
        | Climate Type | {auto_climate} |
        | Power Capacity | {power_mw:.1f} MW |
        | Renewable Share | {renewable_pct_input}% |
        | Renewable Capacity | {power_mw * renewable_ratio:.1f} MW |
        """)

    if run_predict:
        with st.spinner("Running prediction..."):
            try:
                result = predict_new_datacenter(
                    df, selected_country_2, power_mw, renewable_ratio,
                    country_climate_map, model, feature_names,
                )

                st.divider()
                st.markdown('<div class="section-title">⚡ Prediction Results</div>',
                            unsafe_allow_html=True)

                # ── Top KPI row ────────────────────────────────────────
                k1, k2, k3, k4 = st.columns(4)
                risk_icons = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                k1.metric("Total Power Draw (avg MW)",
                          f"{result['predicted_energy_MW']:,.1f} MW",
                          help="IT load × PUE — average continuous draw")
                k2.metric("Annual Energy",
                          f"{result['annual_energy_MWh']:,.0f} MWh/yr",
                          help="Power draw × 8,760 operational hours/yr")
                k3.metric("PUE (Effectiveness)",
                          f"{result['pue']:.2f}",
                          help="Power Usage Effectiveness — 1.0 is perfect, world avg is ~1.55")
                k4.metric("Carbon Footprint",
                          f"{result['annual_carbon_tCO2']:,.0f} tCO₂/yr",
                          help="Based on renewable mix and regional grid intensity")

                st.markdown("<br>", unsafe_allow_html=True)
                res_col1, res_col2 = st.columns([1.2, 1])

                with res_col1:
                    risk = result['risk_level']
                    risk_class = f"risk-{risk.lower()}"

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="pred-unit">Predicted Total Power Consumption</div>
                        <div class="pred-value">{result['predicted_energy_MW']:,.1f}
                            <span class="pred-unit">MW avg draw</span>
                        </div>
                        <div style="font-size:0.88rem;color:#aaa;margin-top:0.3rem;">
                            {result['annual_energy_MWh']:,.0f} MWh/yr &nbsp;·&nbsp;
                            {result['monthly_energy_MWh']:,.0f} MWh/month
                        </div>
                        <hr style="border-color:rgba(255,255,255,0.1); margin:1rem 0;">
                        <div style="display:flex;gap:2rem;">
                          <div>
                            <div style="font-size:0.8rem;color:#aaa;">PUE</div>
                            <div style="font-size:1.4rem;font-weight:700;color:#AB63FA;">{result['pue']:.2f}</div>
                            <div style="font-size:0.75rem;color:#aaa;">{'World-class' if result['pue']<1.2 else 'Efficient' if result['pue']<1.4 else 'Average' if result['pue']<1.6 else 'Inefficient'}</div>
                          </div>
                          <div>
                            <div style="font-size:0.8rem;color:#aaa;">Risk Level</div>
                            <div class="{risk_class}">{risk_icons[risk]} {risk}</div>
                            <div style="font-size:0.75rem;color:#aaa;">{result['risk_explanation'][:60]}...</div>
                          </div>
                        </div>
                        <div class="info-row" style="margin-top:1rem;">
                            <div class="info-pill">🌦️ {result['climate_type']}</div>
                            <div class="info-pill">⚡ {result['power_capacity_MW']:.0f} MW IT Load</div>
                            <div class="info-pill">🌿 {result['renewable_pct']*100:.0f}% Renewable</div>
                            <div class="info-pill">🌍 {result['country']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with res_col2:
                    gauge = risk_gauge_chart(result['risk_level'], result['risk_score'])
                    st.plotly_chart(gauge, use_container_width=True, key="gauge_m2")

                # ── Energy breakdown donut ──────────────────────────────
                st.markdown('<div class="section-title">🌿 Energy Mix Breakdown</div>',
                            unsafe_allow_html=True)
                breakdown_col, tip_col = st.columns([1, 1.5])
                with breakdown_col:
                    st.plotly_chart(energy_breakdown_chart(result),
                                    use_container_width=True, key="donut_m2")
                with tip_col:
                    st.markdown(f"""
                    **📊 Energy Summary**
                    | Metric | Value |
                    |---|---|
                    | IT Load (input) | {result['power_capacity_MW']:.1f} MW |
                    | PUE | {result['pue']:.3f} |
                    | Total avg draw | {result['predicted_energy_MW']:,.1f} MW |
                    | Annual consumption | {result['annual_energy_MWh']:,.0f} MWh/yr |
                    | Monthly consumption | {result['monthly_energy_MWh']:,.0f} MWh/mo |
                    | Renewable energy | {result['renewable_energy_MWh']:,.0f} MWh/yr |
                    | Grid (non-renewable) | {result['non_renewable_energy_MWh']:,.0f} MWh/yr |
                    | Carbon footprint | {result['annual_carbon_tCO2']:,.0f} tCO₂/yr |
                    """)

                # ── Sustainability tips ─────────────────────────────────
                st.markdown('<div class="section-title">💡 Sustainability Recommendations</div>',
                            unsafe_allow_html=True)
                tips_col1, tips_col2 = st.columns(2)
                with tips_col1:
                    if renewable_pct_input < 60:
                        st.warning(f"""
                        **🌿 Increase Renewable Mix** — currently at {renewable_pct_input}%.
                        Pushing above 60% would lower PUE from {result['pue']:.2f} to ~1.25
                        and save ~{(result['pue']-1.25)*power_mw*8760:,.0f} MWh/yr.
                        """)
                    else:
                        st.success(f"✅ Excellent renewable mix ({renewable_pct_input}%)! PUE is {result['pue']:.2f}.")

                with tips_col2:
                    if auto_climate in ['Arid', 'Tropical']:
                        st.warning("""
                        **❄️ Optimise Cooling** — Arid/Tropical climate adds +0.12–0.18 to PUE.
                        Consider liquid cooling, underground design, or free-air cooling corridors.
                        """)
                    else:
                        pue_saving = result['pue'] - 1.12
                        st.success(f"✅ {auto_climate} climate supports efficient cooling. "
                                   f"Optimising to PUE 1.12 could save ~{pue_saving*power_mw*8760:,.0f} MWh/yr.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ── Bottom Banner ────────────────────────────────────────────────────────
import base64 as _b64mod

_pred_banner = os.path.join("graphs", "ai_prediction_banner.png")
if os.path.exists(_pred_banner):
    with open(_pred_banner, "rb") as _f:
        _b64img = _b64mod.b64encode(_f.read()).decode()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    .pred-banner-wrap {
        position: relative; width: 100%;
        border-radius: 18px; overflow: hidden;
        margin-top: 1.8rem;
        box-shadow: 0 12px 40px rgba(171,99,250,0.22), 0 2px 12px rgba(0,0,0,0.4);
    }
    .pred-banner-wrap img {
        width: 100%; height: 200px;
        object-fit: cover; object-position: center 35%;
        display: block;
        filter: brightness(0.60) saturate(1.4);
    }
    .pred-banner-overlay {
        position: absolute; inset: 0;
        background: linear-gradient(
            90deg,
            rgba(13,13,26,0.10) 0%,
            rgba(13,13,26,0.45) 40%,
            rgba(13,13,26,0.85) 100%
        );
        display: flex; flex-direction: column;
        align-items: flex-end; justify-content: center;
        padding: 0 2.5rem;
        border-radius: 18px;
    }
    .pred-banner-title {
        font-size: 1.55rem; font-weight: 800;
        color: #ffffff; margin: 0 0 0.3rem 0;
        text-align: right;
        text-shadow: 0 2px 14px rgba(0,0,0,0.8);
        letter-spacing: -0.3px;
    }
    .pred-banner-sub {
        font-size: 0.88rem; color: rgba(255,255,255,0.75);
        margin: 0 0 0.8rem 0; text-align: right;
        text-shadow: 0 1px 6px rgba(0,0,0,0.6);
        max-width: 420px;
    }
    .pred-stat-row {
        display: flex; gap: 0.6rem; flex-wrap: wrap; justify-content: flex-end;
    }
    .pred-stat-pill {
        background: rgba(171,99,250,0.22);
        border: 1px solid rgba(171,99,250,0.55);
        color: #D8B4FE; font-size: 0.78rem; font-weight: 600;
        padding: 0.22rem 0.8rem; border-radius: 20px;
        backdrop-filter: blur(6px);
    }
    .pred-stat-pill.green {
        background: rgba(0,204,150,0.18);
        border-color: rgba(0,204,150,0.5);
        color: #6EE7B7;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="pred-banner-wrap">
      <img src="data:image/png;base64,{_b64img}" alt="AI Energy Prediction"/>
      <div class="pred-banner-overlay">
        <h2 class="pred-banner-title">🔮 AI-Powered Energy Intelligence</h2>
        <p class="pred-banner-sub">
          Forecasting data center energy demand using compound growth models,
          PUE analytics, and climate-adjusted projections across 191 countries.
        </p>
        <div class="pred-stat-row">
          <span class="pred-stat-pill">📡 191 Countries Modelled</span>
          <span class="pred-stat-pill">⚡ IEA Sector Benchmarks</span>
          <span class="pred-stat-pill green">🌿 PUE × 8,760 hr Formula</span>
          <span class="pred-stat-pill green">📈 +0.7%/yr AI Acceleration</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#555; font-size:0.78rem; margin-top:1rem; padding-bottom:1rem;">
    Models trained on 191 countries &nbsp;·&nbsp; Physics-based PUE Formula &nbsp;·&nbsp; IEA-aligned Growth Projections
</div>
""", unsafe_allow_html=True)
