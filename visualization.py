"""
visualization.py
----------------
All Plotly chart functions for the DC Energy Analytics platform.
Each function accepts a theme-dict `t` for light/dark styling.

# SCALE NOTES (verified from raw data 2024-04):
#   avg_renewable_energy_pct  → 0–100 range  (e.g. 27.0 = 27%)  → display as-is, NO ×100
#   growth_rate_pct           → raw decimal   (e.g. 0.12 = ?)    → median=4.0  ⇒ most stored as float %
#                                values like 0.12 = 0.12%/yr; 4.0 = 4%/yr  → display as-is
#   internet_penetration_pct  → 0–150 range, treated as raw %  → display as-is
#   power_capacity_MW         → raw MW (0–12000)  → display as-is
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

GRAPHS_DIR = "graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

PALETTE = ['#00CC96','#636EFA','#AB63FA','#FFA15A','#19D3F3',
           '#FF6692','#B6E880','#FF97FF','#FECB52','#EF553B']


# ── Shared Plotly layout ───────────────────────────────────────────────
def _base_layout(t, title="", height=420):
    return dict(
        title=dict(text=title, font=dict(size=14, color=t['text'], family='Inter')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=t['text'], size=12),
        margin=dict(l=10, r=10, t=50, b=10),
        height=height,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor=t['border'],
            font=dict(color=t['text'], size=11),
        ),
        xaxis=dict(
            gridcolor=t['divider'], showgrid=True,
            zeroline=False, tickfont=dict(color=t['subtext']),
            title_font=dict(color=t['subtext']),
        ),
        yaxis=dict(
            gridcolor=t['divider'], showgrid=True,
            zeroline=False, tickfont=dict(color=t['subtext']),
            title_font=dict(color=t['subtext']),
        ),
    )


# ══════════════════════════════════════════════════════════════════════
# CHART 1 — Top N by Data Centers  ·  HORIZONTAL GRADIENT BAR
# ══════════════════════════════════════════════════════════════════════
def top_countries_by_datacenters(df, t, n=10):
    top = (df.nlargest(n, 'total_data_centers')
             [['country', 'total_data_centers']]
             .sort_values('total_data_centers'))          # ascending for h-bar
    fig = go.Figure(go.Bar(
        x=top['total_data_centers'],
        y=top['country'],
        orientation='h',
        marker=dict(
            color=top['total_data_centers'],
            colorscale=[[0, '#636EFA'], [0.5, '#00CC96'], [1, '#AB63FA']],
            showscale=False,
            line=dict(width=0),
        ),
        text=top['total_data_centers'].apply(lambda x: f'{int(x):,}'),
        textposition='outside',
        textfont=dict(color=t['text'], size=11),
    ))
    layout = _base_layout(t, f'🏆 Top {n} Countries — Total Data Centers')
    layout['xaxis']['title'] = 'Total Data Centers'
    layout['xaxis']['range'] = [0, top['total_data_centers'].max() * 1.18]
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CHART 2 — Power Capacity  ·  TREEMAP (area = capacity, color = renewable%)
# ══════════════════════════════════════════════════════════════════════
def top_countries_by_power(df, t, n=15):
    top = (df.nlargest(n, 'power_capacity_MW')
             [['country', 'power_capacity_MW', 'avg_renewable_energy_pct']]
             .copy())
    top['label'] = top.apply(
        lambda r: f"{r['country']}<br>{r['power_capacity_MW']:.0f} MW", axis=1
    )
    fig = go.Figure(go.Treemap(
        labels=top['label'],
        parents=['']*len(top),
        values=top['power_capacity_MW'],
        marker=dict(
            colors=top['avg_renewable_energy_pct'],   # already 0-100
            colorscale='RdYlGn',
            cmid=50,
            showscale=True,
            colorbar=dict(title='Renewable %', tickfont=dict(color=t['subtext'])),
            line=dict(width=1, color=t['border']),
        ),
        textfont=dict(color='white', size=11),
        hovertemplate='<b>%{label}</b><br>Power: %{value:.0f} MW<extra></extra>',
    ))
    layout = _base_layout(t,
        f'⚡ Top {n} Countries by Power Capacity — Treemap (color = renewable%)', height=440)
    layout.pop('xaxis', None)
    layout.pop('yaxis', None)
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CHART 3 — Renewable Leaders  ·  VERTICAL COLUMN CHART + threshold
# ══════════════════════════════════════════════════════════════════════
def renewable_leaders_bar(df, t, n=15):
    col = 'avg_renewable_energy_pct'    # already 0-100 scale
    top = (df.nlargest(n, col)
             [['country', col]]
             .sort_values(col, ascending=False)
             .copy())
    fig = go.Figure(go.Bar(
        x=top['country'],
        y=top[col],
        marker=dict(
            color=top[col],
            colorscale=[[0, '#19D3F3'], [0.4, '#00CC96'], [1, '#B6E880']],
            showscale=False, line=dict(width=0),
        ),
        text=top[col].apply(lambda x: f'{x:.0f}%'),
        textposition='outside',
        textfont=dict(color=t['text'], size=10),
        width=0.65,
    ))
    fig.add_hline(y=75, line_dash='dash', line_color='#EF553B',
                  annotation_text='🎯 75% Green Threshold',
                  annotation_font_color='#EF553B', annotation_font_size=11)
    layout = _base_layout(t, f'🌿 Top {n} Renewable Energy Leaders — Vertical Column Chart', height=430)
    layout['yaxis']['title'] = 'Renewable Energy (%)'
    layout['yaxis']['range'] = [0, 118]
    layout['xaxis']['tickangle'] = -35
    layout['xaxis']['showgrid'] = False
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CHART 4 — Fastest Growing  ·  LOLLIPOP DOT CHART
# ══════════════════════════════════════════════════════════════════════
def top_growth_rate_bar(df, t, n=15):
    col = 'growth_rate_pct'             # already raw % (e.g. 5.2 = 5.2%/yr)
    top = (df.nlargest(n, col)
             [['country', col]]
             .sort_values(col, ascending=True)
             .copy())
    median_val = top[col].median()

    fig = go.Figure()
    # Stem lines
    for _, row in top.iterrows():
        clr = '#EF553B' if row[col] >= median_val else '#FFA15A'
        fig.add_shape(type='line',
            x0=0, x1=row[col], y0=row['country'], y1=row['country'],
            line=dict(color=clr, width=2.5))
    # Dots + labels
    fig.add_trace(go.Scatter(
        x=top[col], y=top['country'],
        mode='markers+text',
        text=top[col].apply(lambda x: f' {x:.1f}%/yr'),
        textposition='middle right',
        textfont=dict(color=t['text'], size=10),
        marker=dict(
            size=14,
            color=top[col],
            colorscale=[[0, '#FFA15A'], [1, '#EF553B']],
            line=dict(color=t['border'], width=1),
            showscale=False,
        ),
        hovertemplate='<b>%{y}</b><br>Growth: %{x:.1f}%/yr<extra></extra>',
    ))
    fig.add_vline(x=median_val, line_dash='dot', line_color='#636EFA',
                  annotation_text=f'Median {median_val:.1f}%/yr',
                  annotation_font_color='#636EFA',
                  annotation_position='top right')
    layout = _base_layout(t, f'📈 Top {n} Fastest Growing DC Markets — Lollipop Chart', height=480)
    layout['xaxis']['title'] = 'Annual Growth Rate (%/yr)'
    layout['xaxis']['range'] = [0, top[col].max() * 1.3]
    layout['yaxis']['showgrid'] = False
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CHART 5 — Climate Distribution  ·  DONUT
# ══════════════════════════════════════════════════════════════════════
def climate_distribution_donut(df, t):
    vc = df['Climate_Type'].value_counts().reset_index()
    vc.columns = ['Climate', 'Count']
    fig = go.Figure(go.Pie(
        labels=vc['Climate'], values=vc['Count'],
        hole=0.5,
        marker=dict(colors=PALETTE, line=dict(color='rgba(0,0,0,0)', width=2)),
        textfont=dict(color=t['text'], size=12),
        textinfo='label+percent',
    ))
    fig.update_layout(**_base_layout(t, '🌡️ Countries by Climate Type Distribution'))
    return fig


# ══════════════════════════════════════════════════════════════════════
# CHART 6 — Hyperscale vs Colocation  ·  STACKED H-BAR
# ══════════════════════════════════════════════════════════════════════
def hyperscale_vs_colocation(df, t, n=15):
    top = (df.nlargest(n, 'total_data_centers')
             [['country', 'hyperscale_data_centers', 'colocation_data_centers']]
             .sort_values('hyperscale_data_centers'))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Hyperscale', y=top['country'], x=top['hyperscale_data_centers'],
        orientation='h', marker=dict(color='#636EFA', line=dict(width=0)),
    ))
    fig.add_trace(go.Bar(
        name='Colocation', y=top['country'], x=top['colocation_data_centers'],
        orientation='h', marker=dict(color='#00CC96', line=dict(width=0)),
    ))
    layout = _base_layout(t, f'🏢 Hyperscale vs Colocation DCs — Top {n}', height=460)
    layout['barmode'] = 'stack'
    layout['xaxis']['title'] = 'Number of Data Centers'
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CHART 7 — Renewable vs Non-Renewable  ·  STACKED H-BAR
# ══════════════════════════════════════════════════════════════════════
def renewable_stacked_bar(df, t, n=20):
    top = df.nlargest(n, 'power_capacity_MW').copy()
    # avg_renewable_energy_pct is 0-100, convert to fraction for MW split
    top['renewable_MW']     = top['power_capacity_MW'] * (top['avg_renewable_energy_pct'] / 100)
    top['non_renewable_MW'] = top['power_capacity_MW'] * (1 - top['avg_renewable_energy_pct'] / 100)
    top = top.sort_values('renewable_MW', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='🌿 Renewable', y=top['country'], x=top['renewable_MW'],
        orientation='h', marker=dict(color='#00CC96', line=dict(width=0)),
    ))
    fig.add_trace(go.Bar(
        name='🏭 Non-Renewable', y=top['country'], x=top['non_renewable_MW'],
        orientation='h', marker=dict(color='#EF553B', opacity=0.75, line=dict(width=0)),
    ))
    layout = _base_layout(t, f'♻️ Renewable vs Non-Renewable Power — Top {n} by Capacity', height=520)
    layout['barmode'] = 'stack'
    layout['xaxis']['title'] = 'Power Capacity (MW)'
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CHART 8 — Internet Penetration vs DCs  ·  BUBBLE SCATTER
# (internet_penetration_pct is stored as 0-1 decimal → ×100 for display)
# ══════════════════════════════════════════════════════════════════════
def internet_vs_datacenters_scatter(df, t):
    plot_df = df.copy()
    # internet_penetration_pct: max is ~150 so treat as raw % already (no ×100)
    plot_df['internet_pct'] = plot_df['internet_penetration_pct'].clip(0, 100)
    fig = go.Figure(go.Scatter(
        x=plot_df['internet_pct'],
        y=plot_df['total_data_centers'],
        mode='markers+text',
        text=plot_df['country'],
        textposition='top center',
        textfont=dict(size=8, color=t['subtext']),
        marker=dict(
            size=plot_df['power_capacity_MW'].clip(1, None).apply(
                lambda x: max(6, min(30, x / plot_df['power_capacity_MW'].max() * 30))
            ),
            color=plot_df['avg_renewable_energy_pct'],   # already 0-100
            colorscale='RdYlGn', cmid=50,
            showscale=True,
            colorbar=dict(title='Renewable %', tickfont=dict(color=t['subtext'])),
            line=dict(color=t['border'], width=0.8),
            opacity=0.82,
        ),
        hovertemplate='<b>%{text}</b><br>Internet: %{x:.1f}%<br>DCs: %{y}<extra></extra>',
    ))
    layout = _base_layout(t,
        '🌐 Internet Penetration vs Total Data Centers (bubble=power, color=renewable%)',
        height=500)
    layout['xaxis']['title'] = 'Internet Penetration (%)'
    layout['yaxis']['title'] = 'Total Data Centers'
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CHART 9 — Latency vs DCs  ·  SCATTER (color = growth rate)
# ══════════════════════════════════════════════════════════════════════
def latency_vs_datacenters(df, t):
    fig = go.Figure(go.Scatter(
        x=df['avg_latency_ms'],
        y=df['total_data_centers'],
        mode='markers',
        text=df['country'],
        marker=dict(
            size=9, opacity=0.75,
            color=df['growth_rate_pct'],    # already raw % (e.g. 5.2)
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Growth %/yr', tickfont=dict(color=t['subtext'])),
            line=dict(color=t['border'], width=0.5),
        ),
        hovertemplate='<b>%{text}</b><br>Latency: %{x:.0f} ms<br>DCs: %{y}<extra></extra>',
    ))
    layout = _base_layout(t, '⏱️ Avg Latency vs Total Data Centers (color=growth rate)', height=460)
    layout['xaxis']['title'] = 'Avg Latency to Global Hubs (ms)'
    layout['yaxis']['title'] = 'Total Data Centers'
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CHART 10 — Power Capacity Distribution  ·  HISTOGRAM
# ══════════════════════════════════════════════════════════════════════
def power_capacity_histogram(df, t):
    fig = go.Figure(go.Histogram(
        x=df['power_capacity_MW'], nbinsx=30,
        marker=dict(color='#636EFA', opacity=0.82, line=dict(color=t['border'], width=1)),
    ))
    med = df['power_capacity_MW'].median()
    fig.add_vline(x=med, line_dash='dash', line_color=t['accent'],
                  annotation_text=f'Median: {med:.0f} MW',
                  annotation_font_color=t['accent'])
    layout = _base_layout(t, '📊 Distribution of Power Capacity (MW)')
    layout['xaxis']['title'] = 'Power Capacity (MW)'
    layout['yaxis']['title'] = 'Number of Countries'
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CHART 11 — Renewable % Distribution  ·  HISTOGRAM
# ══════════════════════════════════════════════════════════════════════
def renewable_histogram(df, t):
    col = df['avg_renewable_energy_pct']   # already 0-100
    fig = go.Figure(go.Histogram(
        x=col, nbinsx=25,
        marker=dict(color='#00CC96', opacity=0.82, line=dict(color=t['border'], width=1)),
    ))
    fig.add_vline(x=col.mean(), line_dash='dash', line_color='#FFA15A',
                  annotation_text=f'Mean: {col.mean():.1f}%',
                  annotation_font_color='#FFA15A')
    layout = _base_layout(t, '🌿 Distribution of Renewable Energy Share (%)')
    layout['xaxis']['title'] = 'Renewable Energy (%)'
    layout['yaxis']['title'] = 'Number of Countries'
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CHART 12 — Power by Climate  ·  BOX PLOT
# ══════════════════════════════════════════════════════════════════════
def power_by_climate_box(df, t):
    fig = go.Figure()
    for i, climate in enumerate(sorted(df['Climate_Type'].unique())):
        sub = df[df['Climate_Type'] == climate]['power_capacity_MW']
        fig.add_trace(go.Box(
            y=sub, name=climate,
            marker_color=PALETTE[i % len(PALETTE)],
            line=dict(color=PALETTE[i % len(PALETTE)]),
            boxmean=True,
        ))
    layout = _base_layout(t, '🌡️ Power Capacity by Climate Type', height=440)
    layout['yaxis']['title'] = 'Power Capacity (MW)'
    layout['xaxis']['showgrid'] = False
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CHART 13 — Fiber Connections  ·  HORIZONTAL BAR
# ══════════════════════════════════════════════════════════════════════
def fiber_connections_bar(df, t, n=15):
    top = (df.nlargest(n, 'number_of_fiber_connections')
             [['country', 'number_of_fiber_connections']]
             .sort_values('number_of_fiber_connections'))
    fig = go.Figure(go.Bar(
        x=top['number_of_fiber_connections'], y=top['country'],
        orientation='h',
        marker=dict(
            color=top['number_of_fiber_connections'],
            colorscale='Bluered', showscale=False, line=dict(width=0),
        ),
        text=top['number_of_fiber_connections'].astype(int).astype(str),
        textposition='outside',
        textfont=dict(color=t['text'], size=11),
    ))
    layout = _base_layout(t, f'🔌 Top {n} Countries by Fiber Connections')
    layout['xaxis']['title'] = 'Number of Fiber Connections'
    layout['xaxis']['range'] = [0, top['number_of_fiber_connections'].max() * 1.18]
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CHART 14 — Growth Rate Distribution  ·  HISTOGRAM
# ══════════════════════════════════════════════════════════════════════
def growth_rate_histogram(df, t):
    col = df['growth_rate_pct']    # stored as raw value (e.g. 4.0 = 4%/yr)
    fig = go.Figure(go.Histogram(
        x=col, nbinsx=25,
        marker=dict(color='#FFA15A', opacity=0.85, line=dict(color=t['border'], width=1)),
    ))
    fig.add_vline(x=col.median(), line_dash='dash', line_color='#636EFA',
                  annotation_text=f'Median: {col.median():.2f}%/yr',
                  annotation_font_color='#636EFA')
    layout = _base_layout(t, '📈 Distribution of Annual DC Growth Rate (%/yr)')
    layout['xaxis']['title'] = 'Annual Growth Rate (%/yr)'
    layout['yaxis']['title'] = 'Number of Countries'
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CHART 15 — Feature Correlation  ·  INTERACTIVE HEATMAP
# ══════════════════════════════════════════════════════════════════════
def plotly_correlation_heatmap(df, t):
    num_df = df.select_dtypes(include=np.number).drop(
        columns=['country_encoded'], errors='ignore'
    )
    corr = num_df.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale='RdBu', zmid=0,
        text=corr.values, texttemplate='%{text:.2f}',
        textfont=dict(size=9, color='white'),
        colorbar=dict(tickfont=dict(color=t['subtext'])),
    ))
    layout = _base_layout(t, '🔗 Feature Correlation Matrix', height=500)
    layout['xaxis']['tickangle'] = -35
    layout['xaxis']['showgrid'] = False
    layout['yaxis']['showgrid'] = False
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════
# STATIC — Seaborn Heatmap (PNG save for graphs/)
# ══════════════════════════════════════════════════════════════════════
def seaborn_correlation_heatmap(df, save=True):
    num_df = df.select_dtypes(include=np.number).drop(
        columns=['country_encoded'], errors='ignore'
    )
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    sns.heatmap(corr, annot=True, fmt='.2f', ax=ax, cmap='coolwarm', center=0,
                linewidths=0.5, linecolor='#1e1e2e',
                annot_kws={'size': 8, 'color': 'white'})
    ax.tick_params(colors='#aaa', labelsize=8)
    ax.set_title('Feature Correlation Heatmap', color='#00CC96', fontsize=13, pad=12)
    plt.tight_layout()
    if save:
        path = os.path.join(GRAPHS_DIR, 'correlation_heatmap.png')
        plt.savefig(path, dpi=110, bbox_inches='tight', facecolor='#0d0d1a')
        plt.close()
    return None
