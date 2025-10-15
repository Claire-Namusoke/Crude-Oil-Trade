import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import os
try:
    import openai  # Optional: only used if you enable chat later
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

st.set_page_config(layout="wide")

# Custom CSS (fixed)
st.markdown(
    """
    <style>
    .stApp { background-color: black; }
    [data-testid=\"stSidebar\"] { background-color: black; }
    [data-testid=\"stSidebar\"] > div:first-child { background-color: black; }
    [data-testid=\"stSidebar\"] * { color: white !important; }
    [data-testid=\"stMetricValue\"] { color: #00ffff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌍 Global Crude Oil Trade Dashboard (1995–2021)")

# === Load and clean data ===
DATA_PATH = "C:/Users/clair/Downloads/Global Crude Petroleum Trade 1995-2021.cleaned.csv"
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load data from: {DATA_PATH}\nError: {e}")
    st.info("Please verify the file path or place the CSV at the specified location.")
    st.stop()
df['Continent'] = df['Continent'].astype(str).str.strip()
df['Country'] = df['Country'].astype(str).str.strip()
df['Action'] = df['Action'].astype(str).str.strip().str.title()
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
df['Trade Value'] = pd.to_numeric(df['Trade Value'], errors='coerce')

# === Sidebar filters ===
st.sidebar.header("Filter Options")
all_continents = sorted(df['Continent'].dropna().unique())
selected_years = st.sidebar.slider(
    "Select Year Range",
    int(df['Year'].min()),
    int(df['Year'].max()),
    (int(df['Year'].min()), int(df['Year'].max()))
)

select_all_continents = st.sidebar.checkbox("Select All Continents", value=True)
if select_all_continents:
    selected_continents = all_continents
    st.sidebar.multiselect("Select Continents", all_continents, default=all_continents, disabled=True, key='continent_disabled')
else:
    selected_continents = st.sidebar.multiselect("Select Continents", all_continents, default=[], key='continent_enabled')

# Apply filters
df_year = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
filtered_df = df_year[df_year['Continent'].isin(selected_continents)]

st.caption(f"Active continents: {', '.join(selected_continents)} | Years: {selected_years[0]}–{selected_years[1]}")

# If no data after filtering, show a helpful message and stop rendering further visuals
if filtered_df.empty:
    st.warning("No data matches your current filters. Try selecting more continents or widening the year range.")
    st.stop()

# === OpenAI access setup (hidden) ===
# The sidebar expander for AI access is now hidden from the dashboard.

# === METRICS ===
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    st.markdown(f"**Total Records**<br><span style='color:#00ffff;font-size:1.5em;'>{len(filtered_df):,}</span>", unsafe_allow_html=True)
with col2:
    st.markdown(f"**Total Trade Value**<br><span style='color:#00ffff;font-size:1.5em;'>${filtered_df['Trade Value'].sum()/1e12:.1f}T</span>", unsafe_allow_html=True)
with col3:
    st.markdown(f"**Countries**<br><span style='color:#00ffff;font-size:1.5em;'>{filtered_df['Country'].nunique()}</span>", unsafe_allow_html=True)
with col4:
    years_covered = (filtered_df['Year'].max() - filtered_df['Year'].min() + 1) if not filtered_df.empty else 0
    st.markdown(f"**Years Covered**<br><span style='color:#00ffff;font-size:1.5em;'>{years_covered}</span>", unsafe_allow_html=True)

# Top 10 Countries by Imports and Exports (Table)
# Get top 10 importers
top_importers = (
    filtered_df[filtered_df['Action'] == 'Import']
    .groupby('Country', as_index=False)['Trade Value'].sum()
    .sort_values('Trade Value', ascending=False)
    .head(10)
)
total_imports = filtered_df[filtered_df['Action'] == 'Import']['Trade Value'].sum()
top_importers['Import %'] = (top_importers['Trade Value'] / total_imports * 100).round(2)

# Get top 10 exporters
top_exporters = (
    filtered_df[filtered_df['Action'] == 'Export']
    .groupby('Country', as_index=False)['Trade Value'].sum()
    .sort_values('Trade Value', ascending=False)
    .head(10)
)
total_exports = filtered_df[filtered_df['Action'] == 'Export']['Trade Value'].sum()
top_exporters['Export %'] = (top_exporters['Trade Value'] / total_exports * 100).round(2)

imports_table = top_importers[['Country', 'Import %']].rename(columns={'Country': 'Top Importers'})
exports_table = top_exporters[['Country', 'Export %']].rename(columns={'Country': 'Top Exporters'})
imports_table = imports_table.head(10).copy()
exports_table = exports_table.head(10).copy()
imports_table.index = range(1, 11)
exports_table.index = range(1, 11)

# Gradients
orange_cmap = LinearSegmentedColormap.from_list('orange_gradient', ['#fff5e6', '#ffa600'])
cyan_cmap = LinearSegmentedColormap.from_list('cyan_gradient', ['#e6ffff', '#00ffff'])

st.markdown("**Top 10 Countries by Imports & Exports (%)**")
col_left, col_right = st.columns(2)
with col_left:
    st.markdown("##### Top Importers")
    st.dataframe(
        imports_table.style.format({'Import %': '{:.2f}%'}).background_gradient(cmap=orange_cmap),
        height=350,
        use_container_width=True,
    )
with col_right:
    st.markdown("##### Top Exporters")
    st.dataframe(
        exports_table.style.format({'Export %': '{:.2f}%'}).background_gradient(cmap=cyan_cmap),
        height=350,
        use_container_width=True,
    )

colD1, colD2 = st.columns([1,1])
with colD1:
    continent_action = filtered_df.groupby(['Continent', 'Action'], as_index=False)['Trade Value'].sum()
    continent_action['Trade (Trillion USD)'] = continent_action['Trade Value'] / 1e12
    fig_continent = px.bar(
        continent_action,
        x='Continent',
        y='Trade (Trillion USD)',
        color='Action',
        barmode='group',
        title='Continent Overview: Imports vs Exports',
        color_discrete_map={'Import': '#ffa600', 'Export': '#00ffff'},
        height=350,
    )
    fig_continent.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        title_font=dict(size=16, color='white'),
        legend=dict(orientation='h', y=-0.2, x=0.3),
    )
    st.plotly_chart(fig_continent, use_container_width=True)

with colD2:
    ie_time = filtered_df.groupby(['Year', 'Action'], as_index=False)['Trade Value'].sum()
    years = sorted(ie_time['Year'].dropna().unique().tolist())
    actions = ['Import', 'Export']
    idx = pd.MultiIndex.from_product([years, actions], names=['Year', 'Action'])
    ie_time_full = (
        ie_time.set_index(['Year', 'Action']).reindex(idx, fill_value=0).reset_index().sort_values('Year')
    )
    ie_time_full['Trade (Trillion USD)'] = ie_time_full['Trade Value'] / 1e12
    fig_ie2 = px.line(
        ie_time_full,
        x='Year',
        y='Trade (Trillion USD)',
        color='Action',
        markers=True,
        title='Imports vs Exports (Trillion USD)',
        color_discrete_map={'Import': '#ffa600', 'Export': '#00ffff'},
        category_orders={'Action': ['Import', 'Export']},
        height=350,
    )
    fig_ie2.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        title_font=dict(size=16, color='white'),
        legend=dict(orientation='h', y=-0.2, x=0.3),
    )
    st.plotly_chart(fig_ie2, use_container_width=True)

colA3, colA4 = st.columns([1,1])
with colA3:
    if len(selected_continents) == 1:
        country_trade = filtered_df.groupby('Country', as_index=False)['Trade Value'].sum()
        country_trade = country_trade.sort_values('Trade Value', ascending=True).tail(10)
        country_trade['Trade (Trillion USD)'] = country_trade['Trade Value'] / 1e12
        gradient_colors = [
            mcolors.to_hex(c)
            for c in mcolors.LinearSegmentedColormap.from_list('blue_cyan', ['#001f4d', '#00ffff'])(
                np.linspace(0, 1, len(country_trade))
            )
        ]
        fig2b = px.bar(
            country_trade,
            y='Country',
            x='Trade (Trillion USD)',
            orientation='h',
            height=350,
            title=f'Top 10 Countries by Trade Value - {selected_continents[0]}'
        )
        fig2b.update_traces(
            marker_color=gradient_colors,
            textposition='outside',
            marker_line_color='white',
            marker_line_width=1,
        )
        fig2b.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            title_font=dict(size=16, color='white'),
            xaxis=dict(title='Trade (Trillion USD)', color='white'),
            yaxis=dict(color='white', showticklabels=True),
        )
    else:
        tot_cont = filtered_df.groupby('Continent', as_index=False)['Trade Value'].sum()
        tot_cont = tot_cont.sort_values('Trade Value', ascending=True)
        tot_cont['Trade (Trillion USD)'] = tot_cont['Trade Value'] / 1e12
        gradient_colors = [
            mcolors.to_hex(c)
            for c in mcolors.LinearSegmentedColormap.from_list('blue_cyan', ['#001f4d', '#00ffff'])(
                np.linspace(0, 1, len(tot_cont))
            )
        ]
        fig2b = px.bar(
            tot_cont,
            y='Continent',
            x='Trade (Trillion USD)',
            orientation='h',
            height=350,
            title='Total Trade Value by Continent'
        )
        fig2b.update_traces(
            marker_color=gradient_colors,
            textposition='outside',
            marker_line_color='white',
            marker_line_width=1,
        )
        fig2b.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            title_font=dict(size=16, color='white'),
            xaxis=dict(title='Trade (Trillion USD)', color='white'),
            yaxis=dict(color='white', showticklabels=True),
        )
    st.plotly_chart(fig2b, use_container_width=True)

with colA4:
    continent_scopes = {
        'Africa': 'africa',
        'Asia': 'asia',
        'Europe': 'europe',
        'North America': 'north america',
        'South America': 'south america',
        'Oceania': 'world',
    }
    if len(selected_continents) == 1:
        country_trade = filtered_df.groupby('Country', as_index=False)['Trade Value'].sum()
        country_trade['Trade (Billion USD)'] = country_trade['Trade Value'] / 1e9
        figA4 = px.choropleth(
            country_trade,
            locations="Country",
            locationmode="country names",
            color="Trade (Billion USD)",
            hover_name="Country",
            color_continuous_scale="Viridis",
            title=f"All Trades: {selected_continents[0]} - Country View (Billion USD)",
            height=350,
            scope=continent_scopes.get(selected_continents[0], 'world'),
        )
        figA4.update_layout(
            geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False, showcoastlines=True),
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            title_font=dict(size=16, color='white'),
        )
    else:
        continent_country_trade = filtered_df.groupby(['Continent', 'Country'], as_index=False)['Trade Value'].sum()
        continent_totals = filtered_df.groupby('Continent', as_index=False)['Trade Value'].sum()
        continent_totals.rename(columns={'Trade Value': 'Continent Total'}, inplace=True)
        continent_country_trade = continent_country_trade.merge(continent_totals, on='Continent')
        continent_country_trade['Trade (Billion USD)'] = continent_country_trade['Continent Total'] / 1e9
        figA4 = px.choropleth(
            continent_country_trade,
            locations="Country",
            locationmode="country names",
            color="Trade (Billion USD)",
            hover_name="Continent",
            color_continuous_scale="Viridis",
            title="All Trades: Continent-Level View (Billion USD)",
            height=350,
        )
        figA4.update_layout(
            geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False, showcoastlines=True),
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            title_font=dict(size=16, color='white'),
        )
    st.plotly_chart(figA4, use_container_width=True)

colB1, colB2 = st.columns([1,1])
with colB1:
    yoy = filtered_df.groupby('Year', as_index=False)['Trade Value'].sum().sort_values('Year')
    yoy['Pct Change (%)'] = yoy['Trade Value'].pct_change() * 100
    yoy_plot = yoy.dropna(subset=['Pct Change (%)'])
    colors = np.where(yoy_plot['Pct Change (%)'] >= 0, '#00ffff', '#ff6b6b')
    fig2c = px.bar(yoy_plot, x='Year', y='Pct Change (%)', title='YoY % Change in Trade Value', height=350)
    fig2c.update_traces(marker_color=colors, marker_line_color='white')
    fig2c.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        title_font=dict(size=16, color='white'),
        xaxis=dict(color='white'),
        yaxis=dict(color='white', title='Percent Change (%)'),
    )
    st.plotly_chart(fig2c, use_container_width=True)

with colB2:
    trade_action = filtered_df.groupby('Action', as_index=False)['Trade Value'].sum()
    trade_action['Trade (Trillion USD)'] = trade_action['Trade Value'] / 1e12
    figC1 = px.pie(
        trade_action,
        values='Trade (Trillion USD)',
        names='Action',
        hole=0.55,
        title="Trade Value by Action",
        color='Action',
        color_discrete_map={'Import': '#ffa600', 'Export': '#00ffff'},
        height=350,
    )
    figC1.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        title_font=dict(size=15, color='white'),
        legend=dict(orientation='h', y=-0.2, x=0.15, font=dict(size=10, color='white')),
    )
    st.plotly_chart(figC1, use_container_width=True)

# === Simple Data Q&A (optional GPT rephrasing) ===

# === Langchain Q&A with GPT-4 ===
st.markdown("---")
st.subheader("Ask a question about the data")

import importlib
_langchain_ok = importlib.util.find_spec("langchain") is not None
_lcopenai_ok = importlib.util.find_spec("langchain_openai") is not None

def _get_openai_key():
    try:
        return st.session_state.get('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')
    except Exception:
        return st.session_state.get('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')

if _langchain_ok and _lcopenai_ok and _OPENAI_AVAILABLE:
    from langchain_openai import ChatOpenAI
    from langchain_experimental.agents import create_pandas_dataframe_agent
    import langchain
    # Prefer OpenAI tool-calling agent type; fallback to string if enum not available
    try:
        from langchain.agents import AgentType
        AGENT_TYPE_VALUE = AgentType.OPENAI_TOOLS
    except Exception:
        AGENT_TYPE_VALUE = "openai-tools"
    # st.caption("Ask questions about the data")
    api_key = _get_openai_key()
    if not api_key:
        st.info("Provide your OpenAI API key via environment or session to use Q&A.")
    else:
        # Session state for chat history
        if "qa_history" not in st.session_state:
            st.session_state["qa_history"] = []
        # Region expansions to help the agent understand groupings
        REGION_MAP = {
            "east africa": [
                "Kenya", "Tanzania", "Uganda", "Rwanda", "Burundi", "South Sudan", "Ethiopia"
            ],
            "west africa": [
                "Nigeria", "Ghana", "Cote d'Ivoire", "Ivory Coast", "Senegal", "Benin", "Togo",
                "Sierra Leone", "Liberia", "Guinea", "Guinea-Bissau", "Gambia", "Niger",
                "Burkina Faso", "Mali", "Cape Verde"
            ],
            "north africa": [
                "Egypt", "Libya", "Algeria", "Tunisia", "Morocco", "Sudan"
            ],
            "central africa": [
                "Cameroon", "Chad", "Central African Republic", "Congo",
                "Democratic Republic of the Congo", "Gabon", "Equatorial Guinea"
            ],
            "southern africa": [
                "South Africa", "Botswana", "Namibia", "Zimbabwe", "Zambia", "Mozambique",
                "Angola", "Lesotho", "Eswatini", "Swaziland"
            ],
        }

        def enrich_question(q: str) -> str:
            ql = (q or "").lower()
            notes = []
            for region, countries in REGION_MAP.items():
                if region in ql:
                    # Prefer "Cote d'Ivoire" spelling in our dataset
                    normalized = [c.replace("Ivory Coast", "Cote d'Ivoire") for c in countries]
                    notes.append(f"Treat '{region}' as these countries: {sorted(set(normalized))}.")
            if notes:
                return q + "\n\n" + " ".join(notes)
            return q
        # Chat UI
        # Clear input if flag is set before widget is rendered
        if st.session_state.get("clear_qa_input"):
            st.session_state["qa_input"] = ""
            st.session_state["clear_qa_input"] = False
        question = st.text_input("Type your question (e.g., 'Top exporters in 2010', 'Nigeria imports 2005–2010')", key="qa_input")
        send_clicked = st.button("Send", type="primary")
        if send_clicked:
            if not question.strip():
                st.info("Enter a question first.")
            else:
                # Use full dataframe for agent
                try:
                    llm = ChatOpenAI(api_key=api_key, model="gpt-4o", temperature=0.2)
                    agent = create_pandas_dataframe_agent(
                        llm,
                        df,
                        verbose=False,
                        allow_dangerous_code=True,
                        handle_parsing_errors=True,
                        agent_type=AGENT_TYPE_VALUE,
                        max_iterations=25,
                    )
                    enriched = enrich_question(question)
                    result = agent.invoke({"input": enriched})
                    answer = result.get("output", str(result)) if isinstance(result, dict) else str(result)
                except Exception as e:
                    answer = f"Error from Langchain agent: {e}"
                st.session_state["qa_history"].append((question, answer))
                # Set flag to clear input on next run
                st.session_state["clear_qa_input"] = True
        # Display chat history
        if st.session_state["qa_history"]:
            st.markdown("**Chat History**")
            for q, a in st.session_state["qa_history"][-5:]:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
        if st.button("Clear chat history"):
            st.session_state["qa_history"] = []
else:
    st.info("Langchain and langchain-openai are required for GPT-4 Q&A. Please install them in your environment.")
