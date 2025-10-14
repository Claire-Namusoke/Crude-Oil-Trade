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

# --- Layout Section C ---
colC1, _ = st.columns(2)
with colC1:
    avg_cont = (
        filtered_df.groupby('Continent', as_index=False)['Trade Value'].mean()
        .rename(columns={'Trade Value': 'Average Trade Value (USD)'})
        .sort_values('Average Trade Value (USD)', ascending=False)
    )
    avg_cont['Average (Trillion USD)'] = avg_cont['Average Trade Value (USD)'] / 1e12

    st.subheader("Average Trade Value per Continent")
    st.dataframe(avg_cont[['Continent', 'Average (Trillion USD)']].style.format({'Average (Trillion USD)': '{:.3f}'}))

# --- Optional AI Q&A Section ---
st.write("---")
st.header("Ask Questions About This Data")

with st.sidebar:
    st.subheader("AI Assistant Settings")
    ai_on = st.checkbox("Enable AI Q&A", value=False)

if ai_on:
    try:
        from langchain_openai import ChatOpenAI
        from langchain_experimental.agents import create_pandas_dataframe_agent
        import os
        from dotenv import load_dotenv

        load_dotenv()
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            st.error("OpenAI API key not found. Please add it to your .env file.")

        st.subheader("Ask the data (current filters)")
        question = st.text_input("Type your question about the filtered data")
        if question and openai_key:
            with st.spinner("Thinking..."):
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                agent = create_pandas_dataframe_agent(llm, filtered_df, allow_dangerous_code=True)
                try:
                    answer = agent.run(question)
                    st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"AI error: {e}")
    except Exception as e:
        st.error(f"LangChain not available: {e}")

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
            'Import': '#00ffff',
            'Export': '#ff6b6b'
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

# # --- AI Q&A Section ---
# st.write("---")
# st.header("Ask Questions About This Data")
# st.write("Use the AI below to ask questions and gain insights from your dataset.")

st.header("ASK QUESTIONS ABOUT THIS DATA")
figD2.update_layout(
    coloraxis_colorbar=dict(
        title=dict(text="Avg Duration (min)", font=dict(size=14)),
        thickness=15,  # thinner color bar
        len=0.6,       # shorter color bar
        x=0.95          # move legend slightly right
    ),
    height=650        # increase map height
)

