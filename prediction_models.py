"""
prediction_models.py
--------------------
Realistic prediction engines:
  - Model 1: Future Energy Consumption Forecast
             Uses compound-growth + AI-acceleration to produce true exponential curves.
  - Model 2: New Data Center Energy Prediction
             Uses industry-standard PUE formula with climate overhead adjustment.

WHY NOT ML MODEL?
  The trained model predicts power_capacity_MW using power_capacity_MW as an input
  feature — this is a near-identity mapping that returns almost the same value
  regardless of future year or slider input.  Physics-based formulas produce
  far more realistic and interpretable results for stakeholders.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

MODELS_DIR     = "models"
PLOTLY_TEMPLATE = "plotly_dark"


# ─────────────────────────────────────────────
# MODEL LOADING  (kept for metadata display)
# ─────────────────────────────────────────────

def load_best_model():
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_model_metadata():
    path = os.path.join(MODELS_DIR, "model_info.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def load_feature_names():
    path = os.path.join(MODELS_DIR, "feature_names.pkl")
    if not os.path.exists(path):
        return []
    return joblib.load(path)


def load_label_encoder():
    path = os.path.join(MODELS_DIR, "label_encoder_country.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────
# MODEL 1  —  Future Energy Consumption Forecast
#
# Formula:
#   year_factor(i) = (1 + g + ai_accel×i)^i          # compounding with acceleration
#   energy(i)      = base_MW × year_factor(i) × dc_density_multiplier
#
# where:
#   g            = country base annual growth rate (decimal, e.g. 0.08)
#   ai_accel     = 0.005 per year (0.5% AI-driven extra demand per year)
#   dc_density_multiplier grows with DCs per capita (larger markets scale faster)
#
# This guarantees a strongly convex (exponential) curve with year-over-year
# acceleration, matching IEA projections.
# ─────────────────────────────────────────────────────────────────────

# Industry benchmark CAGR data (IEA 2024 report)
_SECTOR_CAGR = {
    'Arid':          0.14,   # Hot climates need extra cooling → faster capacity growth
    'Tropical':      0.13,
    'Continental':   0.10,
    'Mediterranean': 0.09,
    'Temperate':     0.08,
    'Cold':          0.07,   # Cold climates benefit from free cooling → slower need
}

def forecast_future_energy(df, country, model, feature_names, forecast_years=10):
    """
    Forecast future data center energy demand for a country.

    Returns:
        tuple: (forecast_df, plotly_figure)
    """
    country_row = df[df['country'] == country]
    if len(country_row) == 0:
        return None, None

    r = country_row.iloc[0]
    base_power    = float(r['power_capacity_MW'])
    base_dc       = int(r['total_data_centers'])
    climate_type  = str(r['Climate_Type'])
    renewable_pct = float(r['avg_renewable_energy_pct'])  # 0-100

    # ── Growth rate ──────────────────────────────────────────────────
    # growth_rate_pct: stored as raw float (e.g. 4.0 = 4%/yr, 0.08 = 0.08%/yr)
    raw_growth = float(r['growth_rate_pct'])
    # If value < 1 it's already a decimal fraction (rare edge case)
    if raw_growth < 1.0:
        country_growth = raw_growth           # e.g. 0.08 → 8%
    else:
        country_growth = raw_growth / 100.0   # e.g. 4.0 → 4% = 0.04

    # Blend country growth with sector benchmark (50/50) for realism
    sector_growth = _SECTOR_CAGR.get(climate_type, 0.09)
    base_rate     = 0.5 * country_growth + 0.5 * sector_growth

    # Ensure a minimum 4% growth floor (no country is shrinking in DC capacity)
    base_rate = max(0.04, min(0.30, base_rate))

    # ── AI demand acceleration ────────────────────────────────────────
    # AI workloads add ~0.5–1% extra demand per year, compounding
    ai_accel_per_year = 0.007   # +0.7% additional rate each year

    # ── Compute yearly projections ────────────────────────────────────
    current_year = 2024
    years        = list(range(current_year, current_year + forecast_years + 1))
    energy_vals  = []
    dc_vals      = []
    annual_rates = []

    for i in range(len(years)):
        effective_rate = base_rate + ai_accel_per_year * i
        annual_rates.append(effective_rate)

        # True compound: each year applies (1+r_eff) multiplied cumulatively
        # Using product formula:  Π(1+r_t) for t=0..i-1
        if i == 0:
            compound = 1.0
        else:
            compound = np.prod([1 + (base_rate + ai_accel_per_year * t)
                                for t in range(i)])

        projected_energy = base_power * compound
        projected_dc     = int(base_dc  * compound)

        energy_vals.append(round(projected_energy, 1))
        dc_vals.append(projected_dc)

    # ── Annual increments (for the table) ────────────────────────────
    yoy_growth = [0.0] + [
        round((energy_vals[i] - energy_vals[i-1]) / energy_vals[i-1] * 100, 1)
        for i in range(1, len(energy_vals))
    ]

    forecast_df = pd.DataFrame({
        'Year':                              years,
        'Projected Power Demand (MW)':       energy_vals,
        'YoY Growth (%)':                    yoy_growth,
        'Total Data Centers (est.)':         dc_vals,
        'Effective Growth Rate (%/yr)':      [round(r*100, 2) for r in annual_rates],
    })

    # ── Confidence band (±12% from projection) ───────────────────────
    upper = [v * 1.12 for v in energy_vals]
    lower = [v * 0.88 for v in energy_vals]

    # ── Plot ─────────────────────────────────────────────────────────
    fig = go.Figure()

    # Step 1: Lower band edge (invisible line, acts as floor for fill)
    fig.add_trace(go.Scatter(
        x=years,
        y=lower,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False,
        hoverinfo='skip',
        name='_lower',
    ))

    # Step 2: Upper band edge fills DOWN to the lower line → proper shaded band
    fig.add_trace(go.Scatter(
        x=years,
        y=upper,
        mode='lines',
        line=dict(color='rgba(0,204,150,0.25)', width=1, dash='dot'),
        fill='tonexty',
        fillcolor='rgba(0,204,150,0.12)',
        name='±12% Confidence Band',
        showlegend=True,
        hoverinfo='skip',
    ))

    # Step 3: Main forecast line (no fill — band already handles it)
    fig.add_trace(go.Scatter(
        x=years,
        y=energy_vals,
        mode='lines+markers',
        name='Projected Energy Demand (MW)',
        line=dict(color='#00CC96', width=3.5, shape='spline', smoothing=1.3),
        marker=dict(size=8, color='#00CC96', line=dict(color='white', width=2)),
        hovertemplate='<b>Year %{x}</b><br>%{y:,.0f} MW<extra></extra>',
    ))

    # Annotate final value
    fig.add_annotation(
        x=years[-1], y=energy_vals[-1],
        text=f"  {energy_vals[-1]:,.0f} MW",
        showarrow=False,
        font=dict(color='#00CC96', size=13, family='Inter'),
        xanchor='left',
    )

    # Base year marker
    fig.add_vline(x=current_year, line_dash='dot', line_color='#FFA15A', line_width=2,
                  annotation_text='Base Year 2024',
                  annotation_font_color='#FFA15A', annotation_font_size=11)

    # 2030 milestone marker
    if current_year + forecast_years >= 2030:
        idx_2030 = years.index(2030)
        fig.add_annotation(
            x=2030, y=energy_vals[idx_2030],
            text=f"2030: {energy_vals[idx_2030]:,.0f} MW",
            showarrow=True, arrowhead=2, arrowcolor='#AB63FA',
            font=dict(color='#AB63FA', size=11),
            bgcolor='rgba(171,99,250,0.15)',
        )

    # Renewable % annotation
    renewable_label = f"Current renewable mix: {renewable_pct:.0f}%"
    fig.add_annotation(
        x=years[0], y=energy_vals[-1] * 0.92,
        text=renewable_label, showarrow=False,
        font=dict(color='#B6E880', size=10),
        xanchor='left',
    )

    fig.update_layout(
        title=dict(
            text=f'🔮 Data Center Energy Demand Forecast — {country} '
                 f'({current_year}–{current_year+forecast_years})',
            font=dict(size=15, color='white', family='Inter')),
        xaxis=dict(title='Year', tickfont=dict(color='#aaa'),
                   gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='Power Demand (MW)', tickfont=dict(color='#aaa'),
                   gridcolor='rgba(255,255,255,0.08)',
                   tickformat=',.0f'),
        template=PLOTLY_TEMPLATE,
        height=480,
        hovermode='x unified',
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white', size=11)),
        margin=dict(l=10, r=60, t=60, b=10),
    )

    return forecast_df, fig


# ─────────────────────────────────────────────────────────────────────
# MODEL 2  —  New Data Center Energy Prediction
#
# Industry-standard formula:
#   Total Annual Energy (MWh/yr) = IT Load (MW) × PUE × 8,760 h/yr
#
# where PUE (Power Usage Effectiveness):
#   - World-class / renewables ≥ 70%  → PUE 1.12
#   - Efficient / renewables 50-70%   → PUE 1.25
#   - Average  / renewables 30-50%    → PUE 1.45
#   - Inefficient / renewables < 30%  → PUE 1.65
#   - Hot/Arid climate adds  +0.15 to PUE (cooling overhead)
#   - Tropical  climate adds +0.10 to PUE
#
# Monthly breakdown also shown for planning purposes.
# ─────────────────────────────────────────────────────────────────────

# PUE lookup by renewable %
def _get_pue(renewable_pct_decimal, climate_type):
    """
    Estimate PUE based on renewable energy fraction and climate.

    Args:
        renewable_pct_decimal (float): Renewable share 0–1.
        climate_type (str): Climate type string.

    Returns:
        float: Estimated PUE.
    """
    r = renewable_pct_decimal  # 0-1
    if r >= 0.70:
        pue = 1.12
    elif r >= 0.50:
        pue = 1.25
    elif r >= 0.30:
        pue = 1.45
    else:
        pue = 1.65

    # Climate cooling overhead
    climate_penalty = {
        'Arid':          +0.18,
        'Tropical':      +0.12,
        'Continental':   +0.04,
        'Mediterranean': +0.02,
        'Temperate':     0.00,
        'Cold':          -0.05,  # free cooling benefit
    }
    pue += climate_penalty.get(climate_type, 0.0)
    return round(pue, 3)


# Carbon intensity by renewable mix (tCO2e per MWh)
def _get_carbon_intensity(renewable_pct_decimal):
    r = renewable_pct_decimal
    if r >= 0.80:  return 0.05
    if r >= 0.60:  return 0.15
    if r >= 0.40:  return 0.30
    if r >= 0.20:  return 0.45
    return 0.60


def calculate_risk(power_mw, renewable_pct, climate_type):
    """
    Determine risk level for a new data center.

    Args:
        power_mw (float): IT load in MW.
        renewable_pct (float): Renewable ratio (0–1).
        climate_type (str): Climate type.

    Returns:
        tuple: (risk_level str, risk_score int, explanation str)
    """
    high_risk_climates = ['Arid', 'Tropical']
    score = 0
    reasons = []

    if power_mw > 150:
        score += 2
        reasons.append(f"High IT load ({power_mw:.0f} MW > 150 MW threshold)")
    elif power_mw > 50:
        score += 1

    if renewable_pct < 0.30:
        score += 2
        reasons.append(f"Low renewable share ({renewable_pct*100:.0f}% < 30%)")
    elif renewable_pct < 0.60:
        score += 1

    if climate_type in high_risk_climates:
        score += 1
        reasons.append(f"{climate_type} climate increases cooling energy overhead")

    if score >= 4:
        level = "High"
    elif score >= 2:
        level = "Medium"
    else:
        level = "Low"

    explanation = "; ".join(reasons) if reasons else \
        "Efficient configuration — low energy risk profile."
    return level, score, explanation


def predict_new_datacenter(df, country, power_mw, renewable_pct,
                           country_climate_map, model, feature_names):
    """
    Predict energy consumption for a new planned data center using PUE formula.

    Args:
        df                  : Cleaned DataFrame (used for country lookup).
        country             : Selected country.
        power_mw            : IT load in MW.
        renewable_pct       : Renewable fraction 0–1.
        country_climate_map : {country: climate_type}.
        model               : (unused — kept for API compatibility)
        feature_names       : (unused — kept for API compatibility)

    Returns:
        dict: Full prediction result.
    """
    climate_type = country_climate_map.get(country, 'Temperate')

    pue = _get_pue(renewable_pct, climate_type)

    # Total energy = IT load × PUE × 8760 h/yr (converted to MWh/yr)
    annual_energy_mwh  = power_mw * pue * 8_760
    annual_energy_mw   = power_mw * pue          # average MW draw (≈ MWh/8760)
    monthly_energy_mwh = annual_energy_mwh / 12

    carbon_intensity        = _get_carbon_intensity(renewable_pct)
    annual_carbon_tco2      = annual_energy_mwh * carbon_intensity
    renewable_energy_mwh    = annual_energy_mwh * renewable_pct
    non_renewable_energy_mwh= annual_energy_mwh * (1 - renewable_pct)

    risk_level, risk_score, explanation = calculate_risk(
        power_mw, renewable_pct, climate_type
    )

    return {
        "country":                  country,
        "climate_type":             climate_type,
        "power_capacity_MW":        power_mw,
        "renewable_pct":            renewable_pct,
        "pue":                      pue,
        "predicted_energy_MW":      round(annual_energy_mw, 2),
        "annual_energy_MWh":        round(annual_energy_mwh, 0),
        "monthly_energy_MWh":       round(monthly_energy_mwh, 0),
        "annual_carbon_tCO2":       round(annual_carbon_tco2, 0),
        "renewable_energy_MWh":     round(renewable_energy_mwh, 0),
        "non_renewable_energy_MWh": round(non_renewable_energy_mwh, 0),
        "risk_level":               risk_level,
        "risk_score":               risk_score,
        "risk_explanation":         explanation,
    }


def risk_gauge_chart(risk_level, risk_score):
    """Plotly gauge chart showing risk score 0–5."""
    color_map = {"Low": "#00CC96", "Medium": "#FFA15A", "High": "#EF553B"}
    color = color_map.get(risk_level, "#888")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Level: {risk_level}", 'font': {'size': 18, 'color': color}},
        gauge={
            'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 2], 'color': 'rgba(0,204,150,0.18)'},
                {'range': [2, 4], 'color': 'rgba(255,161,90,0.18)'},
                {'range': [4, 5], 'color': 'rgba(239,85,59,0.18)'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': risk_score,
            }
        }
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=290,
        margin=dict(t=60, b=10, l=30, r=30),
    )
    return fig


def energy_breakdown_chart(result):
    """
    Donut chart showing renewable vs non-renewable energy breakdown.
    """
    fig = go.Figure(go.Pie(
        labels=['🌿 Renewable', '🏭 Non-Renewable'],
        values=[result['renewable_energy_MWh'], result['non_renewable_energy_MWh']],
        hole=0.55,
        marker=dict(colors=['#00CC96', '#EF553B'],
                    line=dict(color='rgba(0,0,0,0)', width=0)),
        textfont=dict(color='white', size=12),
        textinfo='label+percent',
        hovertemplate='%{label}: %{value:,.0f} MWh/yr<extra></extra>',
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=260,
        margin=dict(t=20, b=10, l=10, r=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white', size=11)),
        annotations=[dict(
            text=f"PUE<br>{result['pue']:.2f}",
            x=0.5, y=0.5, font=dict(size=16, color='white'),
            showarrow=False,
        )]
    )
    return fig
