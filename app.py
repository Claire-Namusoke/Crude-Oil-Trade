import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.colors as mcolors

st.title("🌍 Global Crude Oil Trade Dashboard (1995–2021)")

# Load the cleaned data
df = pd.read_csv("C:/Users/clair/Downloads/Global Crude Petroleum Trade 1995-2021.cleaned.csv")

# Normalize key columns
df['Continent'] = df['Continent'].astype(str).str.strip()
df['Country'] = df['Country'].astype(str).str.strip()
df['Action'] = df['Action'].astype(str).str.strip().str.title()
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
df['Trade Value'] = pd.to_numeric(df['Trade Value'], errors='coerce')

# Sidebar filters
st.sidebar.header("Filter Options")
all_continents = sorted(df['Continent'].dropna().unique())

selected_years = st.sidebar.slider(
    "Select Year Range",
    int(df['Year'].min()),
    int(df['Year'].max()),
    (int(df['Year'].min()), int(df['Year'].max()))
)

selected_continents = st.sidebar.multiselect(
    "Select Continents",
    all_continents,
    default=all_continents
)

# Filter data
df_year = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
filtered_df = df_year[df_year['Continent'].isin(selected_continents)]

# Summary banner
st.caption(f"Active continents: {', '.join(selected_continents)} | Years: {selected_years[0]}–{selected_years[1]}")

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"**Total Records**<br><span style='color:#00ffff;font-size:1.5em;'>{len(filtered_df):,}</span>", unsafe_allow_html=True)
with col2:
    st.markdown(f"**Total Trade Value**<br><span style='color:#00ffff;font-size:1.5em;'>${filtered_df['Trade Value'].sum()/1e12:.1f}T</span>", unsafe_allow_html=True)
with col3:
    st.markdown(f"**Countries**<br><span style='color:#00ffff;font-size:1.5em;'>{filtered_df['Country'].nunique()}</span>", unsafe_allow_html=True)
with col4:
    st.markdown(f"**Years Covered**<br><span style='color:#00ffff;font-size:1.5em;'>{filtered_df['Year'].max() - filtered_df['Year'].min() + 1}</span>", unsafe_allow_html=True)

# --- Visual 1: Top 10 Countries by Trade Value ---
top_countries = (
    filtered_df.groupby('Country', as_index=False)['Trade Value'].sum()
    .sort_values('Trade Value', ascending=False)
    .head(10)
)
top_countries['Trade (Trillion USD)'] = top_countries['Trade Value'] / 1e12

# Gradient color
top_countries_sorted = top_countries.sort_values('Trade (Trillion USD)', ascending=False).reset_index(drop=True)
gradient1 = [
    mcolors.to_hex(c)
    for c in mcolors.LinearSegmentedColormap.from_list('blue_cyan', ['#001f4d', '#00ffff'])(np.linspace(0, 1, len(top_countries_sorted)))
]

fig1 = px.bar(
    top_countries_sorted, x='Country', y='Trade (Trillion USD)',
    title='Top 10 Countries by Trade Value',
    height=320
)
fig1.update_traces(marker_color=gradient1, marker_line_color='white', marker_line_width=1)
fig1.update_layout(
    plot_bgcolor='black', paper_bgcolor='black',
    font=dict(color='white'), title_font=dict(size=16, color='white'),
    xaxis=dict(color='white', gridcolor='gray', tickangle=45),
    yaxis=dict(color='white', gridcolor='gray', title='Trade Value (Trillion USD)')
)

# --- Layout Section A ---
colA1, colA2 = st.columns(2)
with colA1:
    st.plotly_chart(fig1, use_container_width=True)

with colA2:
    yearly_trade = filtered_df.groupby("Year", as_index=False)["Trade Value"].sum()
    yearly_trade["Trade Value (Trillion USD)"] = yearly_trade["Trade Value"] / 1e12

    fig2 = px.line(
        yearly_trade, x="Year", y="Trade Value (Trillion USD)",
        markers=True, title="Trade Value Over Time", height=320
    )
    fig2.update_traces(line=dict(color='#00ffff', width=3), marker=dict(color='orange', size=8))
    fig2.update_layout(
        plot_bgcolor='black', paper_bgcolor='black',
        font=dict(color='white'), title_font=dict(size=16, color='white'),
        xaxis=dict(gridcolor='gray', color='white'),
        yaxis=dict(gridcolor='gray', color='white')
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- Layout Section B ---
colB1, colB2 = st.columns(2)
with colB1:
    tot_cont = (
        filtered_df.groupby('Continent')['Trade Value'].sum()
        .reindex(selected_continents, fill_value=0)
        .reset_index()
    )
    tot_cont['Trade (Trillion USD)'] = tot_cont['Trade Value'] / 1e12

    fig2b = px.bar(
        tot_cont, y='Continent', x='Trade (Trillion USD)', orientation='h', height=320
    )
    fig2b.update_traces(marker_color='#00ffff', textposition='outside')
    fig2b.update_layout(
        title='Total Trade Value by Continent',
        plot_bgcolor='black', paper_bgcolor='black',
        font=dict(color='white'), title_font=dict(size=16, color='white'),
        xaxis=dict(title='Trade Value (Trillion USD)', color='white'),
        yaxis=dict(color='white', showticklabels=True)
    )
    st.plotly_chart(fig2b, use_container_width=True)

with colB2:
    yoy = filtered_df.groupby('Year', as_index=False)['Trade Value'].sum().sort_values('Year')
    yoy['Pct Change (%)'] = yoy['Trade Value'].pct_change() * 100
    yoy_plot = yoy.dropna(subset=['Pct Change (%)'])
    colors = np.where(yoy_plot['Pct Change (%)'] >= 0, '#00ffff', '#ff6b6b')

    fig2c = px.bar(yoy_plot, x='Year', y='Pct Change (%)', title='YoY % Change in Trade Value', height=320)
    fig2c.update_traces(marker_color=colors, marker_line_color='white')
    fig2c.update_layout(
        plot_bgcolor='black', paper_bgcolor='black',
        font=dict(color='white'), title_font=dict(size=16, color='white'),
        xaxis=dict(color='white'), yaxis=dict(color='white', title='Percent Change (%)')
    )
    st.plotly_chart(fig2c, use_container_width=True)

# --- Layout Section D: Global Trade Visuals ---
st.write("---")
st.subheader("🌍 Global Trade Insights")

# Create two columns — Donut smaller, Map larger
colD1, colD2 = st.columns([1, 2])  # 1/3 vs 2/3 layout

# --- 1️⃣ Donut Chart: Trade Value by Action ---
with colD1:
    trade_action = (
        filtered_df.groupby('Action', as_index=False)['Trade Value'].sum()
        .sort_values('Trade Value', ascending=False)
    )
    trade_action['Trade (Trillion USD)'] = trade_action['Trade Value'] / 1e12

    figD1 = px.pie(
        trade_action,
        values='Trade (Trillion USD)',
        names='Action',
        hole=0.55,
        title="Trade Value by Action",
        color='Action',
        color_discrete_map={
            'Import': '#ffa600',  # Orange for Imports
            'Export': '#00ffff'   # Cyan for Exports
        }
    )

    figD1.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=13),
        title_font=dict(size=15, color='white'),
        legend=dict(
            title='Action',
            orientation='h',
            y=-0.2, x=0.15,
            font=dict(size=10, color='white')
        )
    )
    st.plotly_chart(figD1, use_container_width=True)

# --- 2️⃣ World Map: Trade Value by Country ---
with colD2:
    country_trade = (
        filtered_df.groupby('Country', as_index=False)['Trade Value'].sum()
    )
    country_trade['Trade (Billion USD)'] = country_trade['Trade Value'] / 1e9

    figD2 = px.choropleth(
        country_trade,
        locations="Country",
        locationmode="country names",
        color="Trade (Billion USD)",
        hover_name="Country",
        color_continuous_scale="Viridis",
        title="All Trades: Country-Level View (Billion USD)",
        height=650
    )

    figD2.update_layout(
        geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False, showcoastlines=True),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        title_font=dict(size=16, color='white'),
        coloraxis_colorbar=dict(
            title=dict(text="Trade (B USD)", font=dict(size=14, color='white')),
            tickfont=dict(color='white'),
            thickness=15,
            len=0.6,
            x=0.95
        )
    )
    st.plotly_chart(figD2, use_container_width=True)

# --- Continent Overview: Imports vs Exports ---
st.write("---")
st.subheader("🌍 Continent Overview: Imports vs Exports")

# Prepare data for continent imports vs exports
continent_action = filtered_df.groupby(['Continent', 'Action'], as_index=False)['Trade Value'].sum()
continent_action['Trade (Trillion USD)'] = continent_action['Trade Value'] / 1e12

# Create grouped bar chart
fig_continent = px.bar(
    continent_action,
    x='Continent',
    y='Trade (Trillion USD)',
    color='Action',
    barmode='group',
    title='Continent Overview: Imports vs Exports',
    color_discrete_map={
        'Import': '#ffa600',   # Orange for Imports
        'Export': '#00ffff'    # Cyan for Exports
    },
    height=320
)

fig_continent.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    title_font=dict(size=16, color='white'),
    xaxis=dict(
        title='Continent',
        color='white',
        gridcolor='gray',
        tickangle=0
    ),
    yaxis=dict(
        title='Trade Value (Trillion USD)',
        color='white',
        gridcolor='gray'
    ),
    legend=dict(
        title='Action',
        orientation='h',
        y=-0.2,
        x=0.3,
        font=dict(size=10, color='white')
    )
)

fig_continent.update_traces(marker_line_color='white', marker_line_width=1)

st.plotly_chart(fig_continent, use_container_width=True)

# --- Imports vs Exports Over Time (2 lines) ---
st.write("---")
st.subheader("📈 Imports vs Exports Over Time")

# Aggregate by Year and Action across the currently filtered continents/countries
ie_time = (
    filtered_df
    .groupby(['Year', 'Action'], as_index=False)['Trade Value']
    .sum()
    .sort_values('Year')
)
ie_time['Trade (Trillion USD)'] = ie_time['Trade Value'] / 1e12

fig_ie2 = px.line(
    ie_time,
    x='Year',
    y='Trade (Trillion USD)',
    color='Action',
    markers=True,
    title='Imports vs Exports (Trillion USD)',
    color_discrete_map={
        'Import': '#ffa600',  # Orange for Imports
        'Export': '#00ffff'   # Cyan for Exports
    },
    category_orders={'Action': ['Import', 'Export']},
    height=420
)

fig_ie2.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    title_font=dict(size=16, color='white'),
    legend=dict(orientation='h', y=-0.2, x=0.3, font=dict(size=10, color='white'))
)
fig_ie2.update_xaxes(color='white', gridcolor='gray', showline=True, linewidth=1, linecolor='white')
fig_ie2.update_yaxes(color='white', gridcolor='gray', showline=True, linewidth=1, linecolor='white')

# --- Imports vs Exports Over Time ---
st.write("---")
st.subheader("📈 Imports vs Exports Over Time")

# Aggregate by Year and Action
ie_time = (
    filtered_df
    .groupby(['Year', 'Action'], as_index=False)['Trade Value']
    .sum()
)

# Ensure both series exist for all years — fill missing with 0
years = sorted(ie_time['Year'].dropna().unique().tolist())
actions = ['Import', 'Export']
idx = pd.MultiIndex.from_product([years, actions], names=['Year', 'Action'])

ie_time_full = (
    ie_time.set_index(['Year', 'Action'])
           .reindex(idx, fill_value=0)
           .reset_index()
           .sort_values('Year')
)

# Scale for readability
ie_time_full['Trade (Trillion USD)'] = ie_time_full['Trade Value'] / 1e12

# Two-line chart
fig_ie = px.line(
    ie_time_full,
    x='Year',
    y='Trade (Trillion USD)',
    color='Action',
    markers=True,
    title='Imports vs Exports (Trillion USD)',
    color_discrete_map={
        'Import': '#ffa600',  # Orange for Imports
        'Export': '#00ffff'   # Cyan for Exports
    },
    category_orders={'Action': ['Import', 'Export']},
    height=420
)

# Dark theme styling to match your dashboard
fig_ie.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    title_font=dict(size=16, color='white'),
    legend=dict(orientation='h', y=-0.2, x=0.3, font=dict(size=10, color='white'))
)
fig_ie.update_xaxes(color='white', gridcolor='gray', showline=True, linewidth=1, linecolor='white')
fig_ie.update_yaxes(color='white', gridcolor='gray', showline=True, linewidth=1, linecolor='white')

st.plotly_chart(fig_ie, use_container_width=True)

