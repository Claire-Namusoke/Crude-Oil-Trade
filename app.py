import streamlit as st
import streamlit as st
from sqlalchemy import create_engine
import pyodbc


# # Database connection parameters
# server_name = 'CLAIRE-NAMUSOKE\\SQLEXPRESS'
# database_name = 'CrudeOilTrade'

# # Create a database engine
# engine = create_engine(f'mssql+pyodbc://{server_name}/{database_name}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes', fast_executemany=True)

# # Test connection
# def test_connection():
#     try:
#         # Connect to the database
#         with engine.connect() as connection:
#             # no-op to ensure the context manager block is not empty;
#             # replace with a lightweight check (e.g., connection.execute("SELECT 1"))
#             pass
#     except Exception as e:
#         st.error(f"Failed to establish connection with the database. Error: {e}")

# # Run the test
# test_connection()

try:
    import plotly.express as px
except Exception:
    st.error("Missing dependency: plotly. Add 'plotly' to requirements.txt and redeploy (or pip install plotly in your environment).")
    st.stop()
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import os
import logging
from logging.handlers import RotatingFileHandler

try:
    import openai
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

st.set_page_config(layout="wide")

# Custom CSS
st.markdown("""
<style>
.stApp { background-color: black; }
[data-testid="stSidebar"] { background-color: black; }
[data-testid="stSidebar"] > div:first-child { background-color: black; }
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stMetricValue"] { color: #00ffff; }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Global Crude Oil Trade Dashboard (1995–2021)")

# === Load and clean data ===
import os
import pandas as pd
import streamlit as st

# Get folder where app.py lives, fallback to cwd if __file__ is not defined
try:
    current_dir = os.path.dirname(__file__)
    if current_dir == '':
        raise Exception
except Exception:
    current_dir = os.getcwd()  # fallback
    

file_name = "Global Crude Petroleum Trade 1995-2021.cleaned.csv"
file_path = os.path.join(current_dir, file_name)

# st.write("Looking for CSV at:", file_path)
# st.write("Files in folder:", os.listdir(current_dir))  # now safe

# Load CSV
try:
    df = pd.read_csv(file_path)
    
except Exception as e:
    st.error(f"Failed to load CSV at: {file_path}\nError: {e}")
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


if filtered_df.empty:
    st.warning("No data matches your current filters. Try selecting more continents or widening the year range.")
    st.stop()

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

# === Top 10 Importers & Exporters ===
top_importers = (filtered_df[filtered_df['Action']=='Import'].groupby('Country', as_index=False)['Trade Value'].sum()
                 .sort_values('Trade Value', ascending=False).head(10))
total_imports = filtered_df[filtered_df['Action']=='Import']['Trade Value'].sum()
top_importers['Import %'] = (top_importers['Trade Value'] / total_imports * 100).round(2)

top_exporters = (filtered_df[filtered_df['Action']=='Export'].groupby('Country', as_index=False)['Trade Value'].sum()
                 .sort_values('Trade Value', ascending=False).head(10))
total_exports = filtered_df[filtered_df['Action']=='Export']['Trade Value'].sum()
top_exporters['Export %'] = (top_exporters['Trade Value'] / total_exports * 100).round(2)

imports_table = top_importers[['Country','Import %']].rename(columns={'Country':'Top Importers'}).copy()
exports_table = top_exporters[['Country','Export %']].rename(columns={'Country':'Top Exporters'}).copy()
imports_table.index = range(1,11)
exports_table.index = range(1,11)

orange_cmap = LinearSegmentedColormap.from_list('orange_gradient', ['#fff5e6', '#ffa600'])
cyan_cmap = LinearSegmentedColormap.from_list('cyan_gradient', ['#e6ffff', '#00ffff'])

st.markdown("**Top 10 Countries by Imports & Exports (%)**")
col_left, col_right = st.columns(2)
with col_left:
    st.markdown("##### Top Importers")
    st.dataframe(imports_table.style.format({'Import %':'{:.2f}%'}).background_gradient(cmap=orange_cmap), height=350, use_container_width=True)
with col_right:
    st.markdown("##### Top Exporters")
    st.dataframe(exports_table.style.format({'Export %':'{:.2f}%'}).background_gradient(cmap=cyan_cmap), height=350, use_container_width=True)

# === Visuals Integration ===
# Continent Overview
colD1, colD2 = st.columns([1,1])
with colD1:
    continent_action = filtered_df.groupby(['Continent','Action'], as_index=False)['Trade Value'].sum()
    continent_action['Trade (Trillion USD)'] = continent_action['Trade Value']/1e12
    fig_continent = px.bar(
        continent_action,
        x='Continent',
        y='Trade (Trillion USD)',
        color='Action',
        barmode='group',
        title='Continent Overview: Imports vs Exports',
        color_discrete_map={'Import':'#ffa600','Export':'#00ffff'},
        height=350
    )
    fig_continent.update_layout(plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'), title_font=dict(size=16,color='white'), legend=dict(orientation='h',y=-0.2,x=0.3))
    st.plotly_chart(fig_continent, use_container_width=True)

with colD2:
    ie_time = filtered_df.groupby(['Year','Action'], as_index=False)['Trade Value'].sum()
    years = sorted(ie_time['Year'].dropna().unique().tolist())
    actions = ['Import','Export']
    idx = pd.MultiIndex.from_product([years, actions], names=['Year','Action'])
    ie_time_full = ie_time.set_index(['Year','Action']).reindex(idx, fill_value=0).reset_index().sort_values('Year')
    ie_time_full['Trade (Trillion USD)'] = ie_time_full['Trade Value']/1e12
    fig_ie2 = px.line(
        ie_time_full,
        x='Year',
        y='Trade (Trillion USD)',
        color='Action',
        markers=True,
        title='Imports vs Exports (Trillion USD)',
        color_discrete_map={'Import':'#ffa600','Export':'#00ffff'},
        category_orders={'Action':['Import','Export']},
        height=350
    )
    fig_ie2.update_layout(plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'), title_font=dict(size=16,color='white'), legend=dict(orientation='h',y=-0.2,x=0.3))
    st.plotly_chart(fig_ie2, use_container_width=True)

# Top Countries / Continent Trade Bar
colA3, colA4 = st.columns([1,1])
with colA3:
    if len(selected_continents)==1:
        country_trade = filtered_df.groupby('Country', as_index=False)['Trade Value'].sum().sort_values('Trade Value', ascending=True).tail(10)
        country_trade['Trade (Trillion USD)'] = country_trade['Trade Value']/1e12
        gradient_colors = [mcolors.to_hex(c) for c in mcolors.LinearSegmentedColormap.from_list('blue_cyan',['#001f4d','#00ffff'])(np.linspace(0,1,len(country_trade)))]
        fig2b = px.bar(country_trade, y='Country', x='Trade (Trillion USD)', orientation='h', height=350, title=f'Top 10 Countries by Trade Value - {selected_continents[0]}')
        fig2b.update_traces(marker_color=gradient_colors,textposition='outside',marker_line_color='white',marker_line_width=1)
        fig2b.update_layout(plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'), title_font=dict(size=16,color='white'), xaxis=dict(title='Trade (Trillion USD)', color='white'), yaxis=dict(color='white'))
    else:
        tot_cont = filtered_df.groupby('Continent', as_index=False)['Trade Value'].sum().sort_values('Trade Value', ascending=True)
        tot_cont['Trade (Trillion USD)'] = tot_cont['Trade Value']/1e12
        gradient_colors = [mcolors.to_hex(c) for c in mcolors.LinearSegmentedColormap.from_list('blue_cyan',['#001f4d','#00ffff'])(np.linspace(0,1,len(tot_cont)))]
        fig2b = px.bar(tot_cont, y='Continent', x='Trade (Trillion USD)', orientation='h', height=350, title='Total Trade Value by Continent')
        fig2b.update_traces(marker_color=gradient_colors,textposition='outside',marker_line_color='white',marker_line_width=1)
        fig2b.update_layout(plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'), title_font=dict(size=16,color='white'), xaxis=dict(title='Trade (Trillion USD)', color='white'), yaxis=dict(color='white'))
    st.plotly_chart(fig2b, use_container_width=True)

# Choropleth Map
with colA4:
    continent_scopes = {'Africa':'africa','Asia':'asia','Europe':'europe','North America':'north america','South America':'south america','Oceania':'world'}
    if len(selected_continents)==1:
        country_trade = filtered_df.groupby('Country', as_index=False)['Trade Value'].sum()
        country_trade['Trade (Billion USD)'] = country_trade['Trade Value']/1e9
        figA4 = px.choropleth(
            country_trade,
            locations="Country",
            locationmode="country names",
            color="Trade (Billion USD)",
            hover_name="Country",
            color_continuous_scale="Viridis",
            title=f"All Trades: {selected_continents[0]} - Country View (Billion USD)",
            height=350,
            scope=continent_scopes.get(selected_continents[0],'world')
        )
        figA4.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)',showframe=False,showcoastlines=True), plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'), title_font=dict(size=16,color='white'))
    else:
        continent_country_trade = filtered_df.groupby(['Continent','Country'], as_index=False)['Trade Value'].sum()
        continent_totals = filtered_df.groupby('Continent', as_index=False)['Trade Value'].sum()
        continent_totals.rename(columns={'Trade Value':'Continent Total'}, inplace=True)
        continent_country_trade = continent_country_trade.merge(continent_totals, on='Continent')
        continent_country_trade['Trade (Billion USD)'] = continent_country_trade['Continent Total']/1e9
        figA4 = px.choropleth(
            continent_country_trade,
            locations="Country",
            locationmode="country names",
            color="Trade (Billion USD)",
            hover_name="Continent",
            color_continuous_scale="Viridis",
            title="All Trades: Continent-Level View (Billion USD)",
            height=350
        )
        figA4.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)',showframe=False,showcoastlines=True), 
                            plot_bgcolor='black', paper_bgcolor='black',font=dict(color='white'), 
                            title_font=dict(size=16,color='white'))
    st.plotly_chart(figA4, use_container_width=True)

# YoY % Change and Pie chart
colB1, colB2 = st.columns([1,1])
with colB1:
    yoy = filtered_df.groupby('Year', as_index=False)['Trade Value'].sum().sort_values('Year')
    yoy['Pct Change (%)'] = yoy['Trade Value'].pct_change()*100
    yoy_plot = yoy.dropna(subset=['Pct Change (%)'])
    colors = np.where(yoy_plot['Pct Change (%)']>=0,'#00ffff','#ff6b6b')
    fig2c = px.bar(yoy_plot, x='Year', y='Pct Change (%)', title='YoY % Change in Trade Value', height=350)
    fig2c.update_traces(marker_color=colors, marker_line_color='white')
    fig2c.update_layout(plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'), title_font=dict(size=16,color='white'), xaxis=dict(color='white'), yaxis=dict(color='white', title='Percent Change (%)'))
    st.plotly_chart(fig2c, use_container_width=True)

with colB2:
    trade_action = filtered_df.groupby('Action', as_index=False)['Trade Value'].sum()
    trade_action['Trade (Trillion USD)'] = trade_action['Trade Value']/1e12
    figC1 = px.pie(trade_action, values='Trade (Trillion USD)', names='Action', hole=0.55, title="Trade Value by Action", color='Action', color_discrete_map={'Import':'#ffa600','Export':'#00ffff'}, height=350)
    figC1.update_layout(plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'), title_font=dict(size=15,color='white'), legend=dict(orientation='h', y=-0.2, x=0.15, font=dict(size=10, color='white')))
    st.plotly_chart(figC1, use_container_width=True)

# === AI Q&A using full dataset ===
st.markdown("---")
st.subheader("Ask a question about the data")

import importlib
_langchain_ok = importlib.util.find_spec("langchain") is not None
_lcopenai_ok = importlib.util.find_spec("langchain_openai") is not None

import os
import streamlit as st

# Path for the .secret.env file 
_ENV_PATH = os.path.join(os.path.dirname(__file__), ".secret.env")
_ENV_VAR = "OPENAI_API_KEY"

def _get_openai_key():
    """
    Returns the OpenAI API key in this order:
    1. st.session_state
    2. Streamlit Secrets (cloud)
    3. Local environment variable
    4. .secret.env file (local, optional)
    """
    key = st.session_state.get(_ENV_VAR)       # 1. session_state
    if not key and st.secrets.get(_ENV_VAR):   # 2. Streamlit Cloud
        key = st.secrets.get(_ENV_VAR)
    if not key and os.getenv(_ENV_VAR):        # 3. local env var
        key = os.getenv(_ENV_VAR)
    if not key:                                # 4. .secret.env file
        key = _read_env_var(_ENV_PATH, _ENV_VAR)
    return key

def _read_env_var(path, var):
    """Read a variable from a .env file."""
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(f"{var}="):
                    return line.split("=", 1)[1]
    except Exception:
        return None
    return None

def _write_env_var(path, var, val):
    """Write or update a variable in a .env file (local use)."""
    try:
        lines = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        found = False
        for i, l in enumerate(lines):
            if l.strip().startswith(f"{var}="):
                lines[i] = f"{var}={val}\n"
                found = True
                break
        if not found:
            lines.append(f"{var}={val}\n")
        dirpath = os.path.dirname(path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        return True
    except Exception:
        return False

def _remove_env_var(path, var):
    """Remove a variable from a .env file (local use)."""
    try:
        if not os.path.exists(path):
            return True
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        new_lines = [l for l in lines if not l.strip().startswith(f"{var}=")]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return True
    except Exception:
        return False


# Continue with LangChain checks and Q&A only if LangChain packages are present and an API key is available via Streamlit secrets / environment / local secrets.toml
def _read_toml_key(paths, var):
    import re
    for path in paths:
        try:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    m = re.match(rf'{re.escape(var)}\s*=\s*["\']?(.*?)["\']?\s*$', line)
                    if m:
                        return m.group(1)
        except Exception:
            continue
    return None

def _get_openai_key():
    try:
        key = None
        try:
            key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            key = None
        if key:
            return key

        # session state
        if st.session_state.get(_ENV_VAR):
            return st.session_state.get(_ENV_VAR)

        # environment
        env_key = os.getenv(_ENV_VAR)
        if env_key:
            return env_key

        # try common local secrets.toml locations (project root, .streamlit/)
        candidate_paths = [
            os.path.join(os.getcwd(), "secrets.toml"),
            os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
            os.path.join(os.path.dirname(__file__), "secrets.toml"),
            os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml"),
            _ENV_PATH
        ]
        toml_key = _read_toml_key(candidate_paths, _ENV_VAR)
        if toml_key:
            return toml_key

        # last resort: read .secret.env style file
        file_key = _read_env_var(_ENV_PATH, _ENV_VAR)
        if file_key:
            return file_key

    except Exception:
        pass
    return None

# Helpers to format numeric values and color AI responses (cyan)
import re
import io

def _human_format(num_str):
    """
    Convert a numeric string or number into a human-friendly compact form.
    Examples:
      1234 -> "1.23K"
      1500000 -> "1.50M"
      2500000000 -> "2.5B"
      1200000000000 -> "1.2T"
      123 -> "123"
      1234.0 -> "1.23K"
      2009 -> "2009"  # preserve years
    """
    try:
        # Accept numeric types directly
        if isinstance(num_str, (int, float)) or (hasattr(num_str, "dtype") and getattr(num_str, "dtype", None) is not None):
            n = float(num_str)
        else:
            s = str(num_str).strip()
            if not s:
                return ""
            # Handle parentheses for negative numbers e.g. (1,234)
            negative = False
            if s.startswith("(") and s.endswith(")"):
                negative = True
                s = s[1:-1].strip()
            # Remove common grouping chars and currency symbols but keep sign and exponent if present
            # Keep digits, dot, minus, plus, and exponent letters
            cleaned = re.sub(r"[^\d\.\-+eE]", "", s)
            if cleaned in ("", ".", "+", "-"):
                return str(num_str)
            n = float(cleaned)
            if negative:
                n = -n
    except Exception:
        return str(num_str)

    absn = abs(n)
    # Preserve four-digit integer-like values (common years) as-is, e.g., 2009 -> "2009"
    if float(n).is_integer() and 1000 <= absn <= 9999:
        return str(int(n))

    # Suffixes in descending order
    units = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]

    for factor, suffix in units:
        if absn >= factor:
            val = n / factor
            # Adaptive decimal places:
            # >=100 -> 0 decimals, >=10 -> 1 decimal, else 2 decimals
            if abs(val) >= 100 or float(val).is_integer():
                formatted = f"{val:,.0f}"
            elif abs(val) >= 10:
                formatted = f"{val:,.1f}"
            else:
                formatted = f"{val:,.2f}"
            return f"{formatted}{suffix}"
    # Values below 1,000: show full number with commas, drop unnecessary .0
    if float(n).is_integer():
        return f"{int(n):,}"
    s = f"{n:,.2f}"
    # strip trailing zeros and possible trailing dot
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s

_number_re = re.compile(r'(?<!\w)(?P<prefix>[$€£]?)\s*(?P<num>\(?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\(?\d+(?:\.\d+)?\)?)(?!\w)')

def _format_numbers_in_text(text):
    def _repl(m):
        prefix = m.group("prefix") or ""
        num = m.group("num")
        formatted = _human_format(num)
        return f"{prefix}{formatted}"
    try:
        return _number_re.sub(_repl, text)
    except Exception:
        return text

def _format_and_color_answer(text):
    if not isinstance(text, str):
        text = str(text)
    formatted = _format_numbers_in_text(text)
    # convert newlines to <br> for HTML display
    html = formatted.replace("\n", "<br>")
    # cyan color for AI responses
    return f"<span style='color:#00ffff'>{html}</span>"

# Ensure the full original dataframe is available to LangChain agents.
# The top-level variable `df` (loaded earlier) contains the full dataset.
# We serialize it to a CSV next to this script so code-run tools inside LangChain
# (or the agent if it executes Python) can read the complete dataset without truncation.
try:
    _FULL_DF_FILENAME = "full_dataset_for_agent.csv"
    _FULL_DF_PATH = os.path.join(os.path.dirname(__file__), _FULL_DF_FILENAME)
    if "df" in globals() and isinstance(df, (pd.DataFrame,)):
        # Write full CSV (overwrite) with UTF-8 encoding
        df.to_csv(_FULL_DF_PATH, index=False, encoding="utf-8")
        # Expose path in session_state for later use
        try:
            st.session_state["_full_df_path"] = _FULL_DF_PATH
        except Exception:
            pass
except Exception:
    # fail silently; this is a best-effort convenience for agent code execution
    pass

def get_full_df_path():
    """Return path to the serialized full dataframe if available, else None."""
    p = None
    try:
        p = st.session_state.get("_full_df_path")
    except Exception:
        p = None
    if p and os.path.exists(p):
        return p
    # fallback: check expected location next to script
    alt = os.path.join(os.path.dirname(__file__), "full_dataset_for_agent.csv")
    if os.path.exists(alt):
        return alt
    return None

def load_full_df():
    """Return the full dataframe object. If the live `df` is present return it,
    otherwise attempt to load the serialized CSV."""
    if "df" in globals() and isinstance(df, pd.DataFrame):
        return df
    p = get_full_df_path()
    if p:
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None

# Optional helper to produce an in-memory CSV string (useful if the agent executes code
# and wants the dataset as a string rather than reading files).
def full_df_to_csv_string():
    d = load_full_df()
    if d is None:
        return None
    buf = io.StringIO()
    d.to_csv(buf, index=False)
    return buf.getvalue()

    # Values below 1,000: show full number with commas, drop unnecessary .0
    if float(n).is_integer():
        return f"{int(n):,}"
    s = f"{n:,.2f}"
    # strip trailing zeros and possible trailing dot
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s

_number_re = re.compile(r'(?<!\w)(?P<prefix>[$€£]?)\s*(?P<num>\(?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\(?\d+(?:\.\d+)?\)?)(?!\w)')

def _format_numbers_in_text(text):
    def _repl(m):
        prefix = m.group("prefix") or ""
        num = m.group("num")
        formatted = _human_format(num)
        return f"{prefix}{formatted}"
    try:
        return _number_re.sub(_repl, text)
    except Exception:
        return text

def _format_and_color_answer(text):
    if not isinstance(text, str):
        text = str(text)
    formatted = _format_numbers_in_text(text)
    # convert newlines to <br> for HTML display
    html = formatted.replace("\n", "<br>")
    # cyan color for AI responses
    return f"<span style='color:#00ffff'>{html}</span>"
    _original_st_checkbox = st.checkbox

    def _hide_pref_checkbox(label, *args, **kwargs):
        try:
            if isinstance(label, str) and "prefer answering with the current ui" in label.lower():
                # Ensure a stable session_state key exists (avoid surprises) and return False (not preferred).
                try:
                    st.session_state.setdefault("prefer_filters_removed", False)
                except Exception:
                    pass
                return False
        except Exception:
            pass
        return _original_st_checkbox(label, *args, **kwargs)

    st.checkbox = _hide_pref_checkbox
    try:
        st.session_state.setdefault("prefer_filters_removed", True)
    except Exception:
        pass

    try:
        st.session_state.setdefault("_last_qa_len", len(st.session_state.get("qa_history", [])))
    except Exception:
        pass

    def _clear_input_if_answered():
        try:
            prev_len = int(st.session_state.get("_last_qa_len", 0))
            cur_len = len(st.session_state.get("qa_history", []))
            if cur_len > prev_len:
                st.session_state["qa_input"] = ""
                st.session_state["qa_triggered"] = False
                st.session_state["_last_qa_len"] = cur_len
        except Exception:
            pass

    # Execute on every rerun so the input is cleared immediately after an answer appears.
    _clear_input_if_answered()

# Gate Q&A on LangChain packages being installed AND an API key being available via secrets/env
if _langchain_ok and _lcopenai_ok:
    api_key = _get_openai_key()
    if not api_key:
        st.info("Provide your OpenAI API key via Streamlit secrets (secrets.toml/.streamlit/secrets.toml), environment variables, or the .secret.env file to enable Q&A.")
    else:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_experimental.agents import create_pandas_dataframe_agent
            import langchain
            try:
                from langchain.agents import AgentType
                AGENT_TYPE_VALUE = AgentType.OPENAI_TOOLS
            except Exception:
                AGENT_TYPE_VALUE = "openai-tools"
            import difflib
            import json
        except Exception as e:
            st.info(f"Langchain packages detected but failed to import runtime classes: {e}")
        else:
            # Ensure the agent has access to both the full dataset and info about the current UI filter.
            try:
                df_agent = df.copy()
                try:
                    df_agent["IN_FILTER"] = df_agent.index.isin(filtered_df.index)
                except Exception:
                    key_cols = ["Country", "Continent", "Year", "Action", "Trade Value"]
                    if all(c in df_agent.columns for c in key_cols):
                        merged = df_agent.merge(
                            filtered_df[key_cols].drop_duplicates(),
                            on=key_cols,
                            how="left",
                            indicator=True
                        )
                        df_agent["IN_FILTER"] = merged["_merge"] == "both"
                    else:
                        df_agent["IN_FILTER"] = False
                agent_csv_path = os.path.join(os.path.dirname(__file__), "agent_dataframe.csv")
                filtered_csv_path = os.path.join(os.path.dirname(__file__), "filtered_dataframe_for_agent.csv")
                try:
                    df_agent.to_csv(agent_csv_path, index=False, encoding="utf-8")
                    filtered_df.to_csv(filtered_csv_path, index=False, encoding="utf-8")
                    st.session_state["_agent_df_path"] = agent_csv_path
                    st.session_state["_filtered_df_path"] = filtered_csv_path
                except Exception:
                    pass
            except Exception:
                df_agent = df.copy()
                df_agent["IN_FILTER"] = False

            SUBREGION_MAP = {
                "east africa": ["Kenya","Tanzania","Uganda","Ethiopia","Somalia","Rwanda","Burundi","Djibouti","Eritrea","South Sudan","Madagascar","Comoros","Seychelles","Mauritius"],
                "west africa": ["Nigeria","Ghana","Senegal","Côte d'Ivoire","Ivory Coast","Mali","Niger","Burkina Faso","Benin","Togo","Sierra Leone","Liberia","Guinea","Guinea-Bissau","The Gambia","Cape Verde"],
                "north africa": ["Egypt","Libya","Tunisia","Algeria","Morocco","Sudan","Mauritania"],
                "southern africa": ["South Africa","Namibia","Botswana","Zimbabwe","Mozambique","Angola","Zambia","Malawi","Lesotho","Swaziland","Eswatini"],
                "central africa": ["Cameroon","Central African Republic","Chad","Republic of the Congo","Democratic Republic of the Congo","Gabon","Equatorial Guinea","São Tomé and Príncipe"],
                "southeast asia": ["Indonesia","Malaysia","Philippines","Thailand","Vietnam","Singapore","Myanmar","Laos","Cambodia","Brunei","Timor-Leste"],
                "south asia": ["India","Pakistan","Bangladesh","Sri Lanka","Nepal","Bhutan","Maldives","Afghanistan"],
                "central america": ["Guatemala","Belize","Honduras","El Salvador","Nicaragua","Costa Rica","Panama"],
                "caribbean": ["Cuba","Jamaica","Haiti","Dominican Republic","Bahamas","Barbados","Trinidad and Tobago"],
            }

            def _detect_subregion_countries(query):
                q = query.lower()
                matched = {}
                for key, countries in SUBREGION_MAP.items():
                    if key in q:
                        matched[key] = countries
                keys = list(SUBREGION_MAP.keys())
                close = difflib.get_close_matches(q, keys, n=1, cutoff=0.75)
                for c in close:
                    matched[c] = SUBREGION_MAP[c]
                return matched

            if "qa_history" not in st.session_state:
                st.session_state["qa_history"] = []

            # Enter triggers submission
            question = st.text_input(
                "Type your question (press Enter to submit)",
                key="qa_input",
                on_change=lambda: st.session_state.update({"qa_triggered": True})
            )

            if st.session_state.get("qa_triggered"):
                st.session_state["qa_triggered"] = False
                if question and question.strip():
                    user_question = question.strip()

                    # Preprocess question to help the agent understand context and subregions.
                    augmented_question = user_question + "\n\nPlease answer concisely in plain language. Use compact numeric units (K/M/B/T) for large numbers. If the user requests answers limited to the current UI selection, prefer rows where IN_FILTER == True, but do not mention or justify filter usage in the answer."

                    context_lines = [
                        "Context for the agent:",
                        "- You have access to a pandas DataFrame where rows currently selected in the UI are marked with a boolean column named 'IN_FILTER'.",
                        f"- Full dataframe CSV path (for code-run tools): {st.session_state.get('_agent_df_path', agent_csv_path)}",
                        f"- Filtered (UI) dataframe CSV path (for code-run tools): {st.session_state.get('_filtered_df_path', filtered_csv_path)}",
                        "- If the user requests answers limited to the current UI selection, prefer rows with IN_FILTER == True, but do NOT include any explanatory text about filters in your answer."
                    ]
                    augmented_question = augmented_question + "\n\n" + "\n".join(context_lines)

                    subregions = _detect_subregion_countries(user_question)
                    if subregions:
                        for key, countries in subregions.items():
                            augmented_question += f"\n\nNote: Interpret '{key}' to include these countries (best-effort): {', '.join(countries)}."

                    try:
                        llm = ChatOpenAI(api_key=api_key, model="gpt-4o", temperature=0.0)
                        agent = create_pandas_dataframe_agent(
                            llm,
                            df_agent,
                            verbose=False,
                            allow_dangerous_code=True,
                            handle_parsing_errors=True,
                            agent_type=AGENT_TYPE_VALUE,
                            max_iterations=25
                        )

                        # Request full-dataset analysis in a concise form (agent may use IN_FILTER if user asked)
                        augmented_question = (
                            "Use all available rows in the dataset to compute and support your answer. "
                            "Do not state, reveal, or describe that you consulted the entire dataset or dataframe; "
                            "simply present the concise results and reasoning.\n\n"
                        ) + augmented_question

                        result = agent.invoke({"input": augmented_question})
                        answer = result.get("output", str(result)) if isinstance(result, dict) else str(result)
                    except Exception as e:
                        answer = f"Error from Langchain agent: {e}"
                    finally:
                        # Ensure the input is cleared from the UI after processing (works on success or error)
                        try:
                            st.session_state["qa_input"] = ""
                        except Exception:
                            pass
                        try:
                            st.session_state["qa_triggered"] = False
                        except Exception:
                            pass

                    # store raw answer; display will be formatted and colored
                    st.session_state["qa_history"].append((user_question, answer))

                    # Always clear the input and update the QA counter immediately after appending the answer.
                    try:
                        st.session_state["qa_input"] = ""
                    except Exception:
                        pass
                    try:
                        st.session_state["qa_triggered"] = False
                    except Exception:
                        pass
                    try:
                        st.session_state["_last_qa_len"] = len(st.session_state.get("qa_history", []))
                    except Exception:
                        pass

                    try:
                        # Force a quick rerun so the input text box shows empty to the user immediately.
                        st.experimental_rerun()
                    except Exception:
                        # If rerun isn't allowed in this environment, the earlier session_state clears will still take effect on the next render.
                        pass

            if st.session_state["qa_history"]:
                st.markdown("**Chat History**")
                for q, a in st.session_state["qa_history"][-10:]:
                    st.markdown(f"**Q:** {q}")
                    colored = _format_and_color_answer(a)
                    st.markdown(f"**A:** {colored}", unsafe_allow_html=True)

            if st.button("Clear chat history"):
                st.session_state["qa_history"] = []
else:
    st.info(
        "Langchain and langchain-openai packages are required for GPT Q&A. "
        "Install them in your environment and provide an OPENAI_API_KEY in "
        "Streamlit secrets or environment variables."
    )

# === Logging setup ===
LOG_PATH = os.path.join(os.path.dirname(__file__), "app.log")
root_logger = logging.getLogger()
# If no file or rotating handler is present, add one and ensure a console handler exists.
if not any(isinstance(h, (RotatingFileHandler, logging.FileHandler)) for h in root_logger.handlers):
    log_dir = os.path.dirname(LOG_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    file_handler = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

# Ensure at least a stream handler is present for console output.
if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    try:
        stream_handler = logging.StreamHandler()
        # reuse formatter if set above, otherwise create one
        if 'formatter' not in locals():
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        root_logger.addHandler(stream_handler)
    except Exception:
        pass

root_logger.setLevel(logging.INFO)

import os

# Get the API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OpenAI API key not found. Please set it as an environment variable.")
else:
    # Example: initialize OpenAI client
    import openai
    openai.api_key = api_key

