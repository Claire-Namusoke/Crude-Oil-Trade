import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

st.title("Crude Oil Trade Data Dashboard")

# Load the cleaned data
df = pd.read_csv("C:/Users/clair/Downloads/Global Crude Petroleum Trade 1995-2021.cleaned.csv")

all_continents = sorted(df['Continent'].dropna().unique())
st.sidebar.header("Filter Options")
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
filtered_df = df[(df['Year'] >= selected_years[0]) & 
                (df['Year'] <= selected_years[1]) & 
                (df['Continent'].isin(selected_continents))]

# Display summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", f"{len(filtered_df):,}")
with col2:
    st.metric("Total Trade Value", f"${filtered_df['Trade Value'].sum()/1e12:.1f}T")
with col3:
    st.metric("Countries", f"{filtered_df['Country'].nunique()}")
with col4:
    st.metric("Years Covered", f"{filtered_df['Year'].max() - filtered_df['Year'].min() + 1}")

# 1. Top 10 Countries by Trade Value
st.subheader("Top 10 Countries by Trade Value")
top_countries = filtered_df.groupby("Country")["Trade Value"].sum().nlargest(10)
fig1, ax1 = plt.subplots(figsize=(12, 6))
bars = ax1.bar(range(len(top_countries)), top_countries.values / 1e12, color='darkblue', alpha=0.7)
ax1.set_title('Top 10 Countries by Trade Value', fontsize=16, fontweight='bold')
ax1.set_xlabel('Country')
ax1.set_ylabel('Trade Value (Trillion USD)')
ax1.set_xticks(range(len(top_countries)))
ax1.set_xticklabels(top_countries.index, rotation=45, ha='right')
for i, v in enumerate(top_countries.values / 1e12):
    ax1.text(i, v + 0.05, f'{v:.1f}T', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
st.pyplot(fig1)

# 2. Trade Value Over Time
st.subheader("Trade Value Over Time")
yearly_trade = filtered_df.groupby("Year")["Trade Value"].sum()
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(yearly_trade.index, yearly_trade.values / 1e12, marker='o', linewidth=3, markersize=6, color='red')
ax2.set_title('Trade Value Over Time', fontsize=16, fontweight='bold')
ax2.set_xlabel('Year')
ax2.set_ylabel('Trade Value (Trillion USD)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)

# 3. Trade by Action (Import/Export)
st.subheader("Trade Value by Action (Import/Export)")
action_trade = filtered_df.groupby("Action")["Trade Value"].sum()
fig3, ax3 = plt.subplots(figsize=(8, 6))
colors = ['lightcoral', 'lightblue']
wedges, texts, autotexts = ax3.pie(action_trade.values, labels=action_trade.index, autopct='%1.1f%%', 
                                  colors=colors, startangle=90)
ax3.set_title('Trade Value by Action', fontsize=16, fontweight='bold')
for autotext in autotexts:
    autotext.set_fontweight('bold')
plt.tight_layout()
st.pyplot(fig3)

# 4. Trade by Continent
st.subheader("Trade Value by Continent")
continent_trade = filtered_df.groupby("Continent")["Trade Value"].sum().sort_values(ascending=True)
fig4, ax4 = plt.subplots(figsize=(12, 6))
bars = ax4.barh(continent_trade.index, continent_trade.values / 1e12, color='green', alpha=0.7)
ax4.set_title('Trade Value by Continent', fontsize=16, fontweight='bold')
ax4.set_xlabel('Trade Value (Trillion USD)')
for i, v in enumerate(continent_trade.values / 1e12):
    ax4.text(v + 0.2, i, f'{v:.1f}T', va='center', fontweight='bold')
plt.tight_layout()
st.pyplot(fig4)

# 5. Top Exporters
st.subheader("Top 10 Exporters")
exporters = filtered_df[filtered_df["Action"].str.lower() == "export"].groupby("Country")["Trade Value"].sum().nlargest(10)
fig5, ax5 = plt.subplots(figsize=(12, 6))
bars = ax5.bar(range(len(exporters)), exporters.values / 1e12, color='orange', alpha=0.8)
ax5.set_title('Top 10 Exporting Countries', fontsize=16, fontweight='bold')
ax5.set_xlabel('Country')
ax5.set_ylabel('Export Value (Trillion USD)')
ax5.set_xticks(range(len(exporters)))
ax5.set_xticklabels(exporters.index, rotation=45, ha='right')
for i, v in enumerate(exporters.values / 1e12):
    ax5.text(i, v + 0.05, f'{v:.1f}T', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
st.pyplot(fig5)

# Display raw data option
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.dataframe(filtered_df)

# ==========================
# Interactive Maps section 🌍
# ==========================
st.write("---")
st.header("Maps: Imports and Exports by Continent and Country")

# Choose action to visualize
action_choice = st.radio(
    "Select trade action to visualize",
    ["All", "Export", "Import"],
    horizontal=True,
)

if action_choice == "All":
    df_action = filtered_df.copy()
else:
    df_action = filtered_df[filtered_df["Action"].str.lower() == action_choice.lower()]

# Aggregate by continent for overview
continent_totals = (
    df_action.groupby("Continent")["Trade Value"].sum().reset_index().sort_values("Trade Value", ascending=False)
)

import pycountry
# Aggregate by country for country-level map
country_totals = (
    df_action.groupby(["Country", "Continent"])["Trade Value"].sum().reset_index()
)
# Add ISO-3 country codes
def get_iso3(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except Exception:
        return None
country_totals["ISO3"] = country_totals["Country"].apply(get_iso3)
# Merge continent totals back onto country data so we can color by continent value if desired
country_with_cont_total = country_totals.merge(
    continent_totals.rename(columns={"Trade Value": "Continent Total"}), on="Continent", how="left"
)

# Continent drill-down selector
drill_options = ["All"] + list(continent_totals["Continent"].unique())
drill_cont = st.selectbox("Drill down by continent (optional)", drill_options)

if drill_cont != "All":
    map_df = country_with_cont_total[country_with_cont_total["Continent"] == drill_cont]
else:
    map_df = country_with_cont_total

st.subheader("Continent overview (stacked imports vs exports)")
# Stacked bar per continent: Imports vs Exports
cont_ie = (
    filtered_df.groupby(["Continent", filtered_df["Action"].str.title()])["Trade Value"].sum()
    .unstack(fill_value=0)
    .reset_index()
)
if "Export" not in cont_ie.columns:
    cont_ie["Export"] = 0
if "Import" not in cont_ie.columns:
    cont_ie["Import"] = 0
fig_cont_bar = px.bar(
    cont_ie,
    x="Continent",
    y=["Export", "Import"],
    title="Trade Value by Continent (Stacked)",
    labels={"value": "Trade Value (USD)", "variable": "Action"},
    barmode="stack",
)
fig_cont_bar.update_layout(legend_title_text="Action")
st.plotly_chart(fig_cont_bar, use_container_width=True)

st.subheader("World map (hover to see values)")
# If user chose All, color by continent total for a continent-themed map; otherwise by country totals
color_col = "Trade Value" if action_choice != "All" else "Continent Total"
title_suffix = action_choice if action_choice != "All" else "All Trades"
fig_country_map = px.choropleth(
    map_df,
    locations="ISO3",
    locationmode="ISO-3",
    color=color_col,
    hover_name="Country",
    hover_data={
        "Continent": True,
        "Trade Value": ":,",
        "Continent Total": ":,",
    },
    color_continuous_scale="Viridis",
    title=f"{title_suffix}: Country-level view",
)
fig_country_map.update_layout(margin=dict(l=0, r=0, t=60, b=0))
st.plotly_chart(fig_country_map, use_container_width=True)

# If a continent is selected, show country list and a detailed country-only map
if drill_cont != "All":
    st.subheader(f"Countries in {drill_cont}")
    # Show table of countries and values for the chosen continent
    st.dataframe(
        map_df[["Country", "Trade Value"]].sort_values("Trade Value", ascending=False).rename(
            columns={"Trade Value": "Trade Value (USD)"}
        )
    )

