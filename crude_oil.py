"""Utilities to load, clean, and save the Global Crude Petroleum Trade 1995-2021 dataset."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def load_crude_trade(path: str) -> pd.DataFrame:
	"""Load a CSV/Excel/Parquet/Feather file at `path` into a DataFrame."""
	if os.path.exists(path):
		_, ext = os.path.splitext(path)
		ext = ext.lower()
		if ext in (".xlsx", ".xls"):
			return pd.read_excel(path)
		if ext == ".csv" or ext == "":
			return pd.read_csv(path)
		if ext == ".parquet":
			return pd.read_parquet(path)
		if ext == ".feather":
			return pd.read_feather(path)
	# Try adding typical extensions if path has no extension
	root, ext = os.path.splitext(path)
	for e in (".csv", ".xlsx", ".xls", ".parquet", ".feather"):
		candidate = root + e
		if os.path.exists(candidate):
			if e == ".csv":
				return pd.read_csv(candidate)
			if e in (".xlsx", ".xls"):
				return pd.read_excel(candidate)
			if e == ".parquet":
				return pd.read_parquet(candidate)
			if e == ".feather":
				return pd.read_feather(candidate)
	raise FileNotFoundError(f"File not found: {path}")

def clean_crude_trade(df: pd.DataFrame) -> pd.DataFrame:
	"""Basic cleaning: strip whitespace, normalize columns, convert types, drop missing."""
	df = df.copy()
	df.columns = [str(c).strip() for c in df.columns]
	obj_cols = df.select_dtypes(include=[object]).columns
	for c in obj_cols:
		df[c] = df[c].astype(str).str.strip()
	col_map = {}
	for c in df.columns:
		low = c.lower()
		if low in ("trade value", "trade_value", "value"):
			col_map[c] = "Trade Value"
		if low == "country":
			col_map[c] = "Country"
		if low == "continent":
			col_map[c] = "Continent"
		if low == "year":
			col_map[c] = "Year"
		if low in ("action", "type"):
			col_map[c] = "Action"
	df = df.rename(columns=col_map)
	if "Trade Value" in df.columns:
		df["Trade Value"] = pd.to_numeric(df["Trade Value"], errors="coerce").fillna(0)
	if "Year" in df.columns:
		df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
	if "Country" in df.columns and "Year" in df.columns:
		df = df[df["Country"].notna() & df["Year"].notna()]
	df = df.reset_index(drop=True)
	return df

def save_cleaned(df: pd.DataFrame, out_dir: str, basename: str) -> tuple[str, str]:
	"""Save cleaned DataFrame to CSV and Parquet in out_dir using basename."""
	os.makedirs(out_dir, exist_ok=True)
	csv_path = os.path.join(out_dir, basename + ".cleaned.csv")
	parquet_path = os.path.join(out_dir, basename + ".cleaned.parquet")
	df.to_csv(csv_path, index=False)
	try:
		df.to_parquet(parquet_path, index=False)
	except Exception:
		raise
	return csv_path, parquet_path

if __name__ == "__main__":

	src = r"C:\Users\clair\Downloads\Global Crude Petroleum Trade 1995-2021.csv"
	out_dir = r"C:\Users\clair\Downloads"
	basename = "Global Crude Petroleum Trade 1995-2021"
	print("Loading:", src)
	df = load_crude_trade(src)
	print("Original shape:", df.shape)
	df_clean = clean_crude_trade(df)
	print("Cleaned shape:", df_clean.shape)
	# Add ID column starting from 1
	df_clean.insert(0, "ID", range(1, len(df_clean) + 1))
	print("\nTable with ID column (first 10 rows):")
	print(df_clean.head(10).to_string(index=False))
	csv_out, parquet_out = save_cleaned(df_clean, out_dir, basename)
	print("Saved cleaned CSV:", csv_out)
	print("Saved cleaned Parquet:", parquet_out)

	# Count Europe exports in 2020
	europe_2020_exports = df_clean[(df_clean["Continent"].str.lower() == "europe") & (df_clean["Year"] == 2020) & (df_clean["Action"].str.lower() == "export")]
	print(f"\nEurope export count in 2020: {len(europe_2020_exports)}")

# Calculate yearly trade values and percentage changes
	yearly_trade = df_clean.groupby("Year")["Trade Value"].sum().reset_index()
	yearly_trade["Pct_Change"] = yearly_trade["Trade Value"].pct_change() * 100

	print("\nYearly Trade Value Changes:")
	print("Year    Trade Value      % Change")
	print("-" * 40)
	for _, row in yearly_trade.iterrows():
		pct = f"{row['Pct_Change']:>7.2f}%" if pd.notna(row['Pct_Change']) else "    N/A"
		print(f"{int(row['Year']):4d}  {row['Trade Value']:>13,.0f}  {pct}")

	# Total trade value
	total_trade_value = df_clean["Trade Value"].sum() if "Trade Value" in df_clean.columns else 0
	print(f"\nTotal Trade Value: {total_trade_value}")

	# Top 10 countries by trade value
	if "Country" in df_clean.columns and "Trade Value" in df_clean.columns:
		top_countries = df_clean.groupby("Country")["Trade Value"].sum().nlargest(10)
		print("\nTop 10 countries by Trade Value:")
		print(top_countries)
		
		# Create bar chart for top 10 countries
		# Create bar chart for top 10 countries with a grey gradient (darker -> higher values)
		# Set figure and axes background to light greys for better contrast
		fig, ax = plt.subplots(figsize=(12, 6), facecolor='#f2f2f2')  # very light grey figure background
		# Prepare values (in trillions for plotting) and normalize for colormap
		values = top_countries.values.astype(float)
		values_trill = values / 1e12
		norm = plt.Normalize(vmin=values_trill.min(), vmax=values_trill.max())
		# Override colors: make all bars solid black with white edge for clarity
		ax.set_facecolor('#e0e0e0')  # slightly darker axes background for better contrast
		colors = ['black'] * len(values_trill)
		bars = ax.bar(top_countries.index, values_trill, color=colors, edgecolor='white', linewidth=1)
		ax.set_title('Top 10 Countries by Trade Value')
		ax.set_xlabel('Country')
		ax.set_ylabel('Trade Value (Trillion USD)')
		plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
		
		# Add value labels on top of each bar (showing trillions)
		# Labels are white on the black bars for visibility
		for bar, val in zip(bars, values_trill):
			height = val
			ax.text(bar.get_x() + bar.get_width()/2., height,
					f'{height:.2f}T',
					color='white', ha='center', va='bottom')
		
		plt.tight_layout()
		# Save with the figure facecolor so background remains light grey
		fig.savefig('top_10_countries_trade_value.png', facecolor=fig.get_facecolor())
		plt.close(fig)
		print("\nBar chart has been saved as 'top_10_countries_trade_value.png'")

	# Change data type for Trade Value to integer
	if "Trade Value" in df_clean.columns:
		df_clean["Trade Value"] = df_clean["Trade Value"].astype(int)
		print("\nData types after conversion:")
		print(df_clean.dtypes)

	# Top 10 Leading exporting countries
	if "Country" in df_clean.columns and "Action" in df_clean.columns and "Trade Value" in df_clean.columns:
		top_exporters = df_clean[df_clean["Action"].str.lower() == "export"].groupby("Country")["Trade Value"].sum().nlargest(10)
		print("\nTop 10 Leading Exporting Countries:")
		print(top_exporters)
	
    # Average trade value per year
	if "Year" in df_clean.columns and "Trade Value" in df_clean.columns:
		avg_trade_per_year = df_clean.groupby("Year")["Trade Value"].mean()
		print("\nAverage Trade Value per Year:")
		print("Year    Average Trade Value")
		print("-" * 40)
		for year, value in avg_trade_per_year.items():
			print(f"{int(year):4d}  {value:>18,.2f}")

	st.title("Crude Oil Trade Data Explorer")

	# Load cleaned data
	df = pd.read_csv("C:/Users/clair/Downloads/Global Crude Petroleum Trade 1995-2021.cleaned.csv")

	st.write("## Data Table")
	st.dataframe(df)

	# Total trade value for all years
	total_trade_value = df["Trade Value"].sum()
	st.metric("Total Trade Value (USD)", f"{total_trade_value:,.0f}")

	# Yearly trade value bar chart
	if "Year" in df.columns and "Trade Value" in df.columns:
	    yearly_trade = df.groupby("Year")["Trade Value"].sum()
st.write("## Yearly Trade Value")
st.bar_chart(yearly_trade)

st.write("---")
st.write("Select columns, filter, and use Streamlit's built-in visualization options from the UI.")

"""
Comprehensive visualization of the crude oil trade data
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the cleaned data
df = pd.read_csv("C:/Users/clair/Downloads/Global Crude Petroleum Trade 1995-2021.cleaned.csv")

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. Top 10 Countries by Trade Value
plt.subplot(2, 3, 1)
top_countries = df.groupby("Country")["Trade Value"].sum().nlargest(10)
bars = plt.bar(range(len(top_countries)), top_countries.values / 1e12, color='darkblue')
plt.title('Top 10 Countries by Trade Value', fontsize=14, fontweight='bold')
plt.xlabel('Country')
plt.ylabel('Trade Value (Trillion USD)')
plt.xticks(range(len(top_countries)), top_countries.index, rotation=45, ha='right')
for i, v in enumerate(top_countries.values / 1e12):
    plt.text(i, v + 0.05, f'{v:.1f}T', ha='center', va='bottom', fontweight='bold')

# 2. Trade Value by Year
plt.subplot(2, 3, 2)
yearly_trade = df.groupby("Year")["Trade Value"].sum()
plt.plot(yearly_trade.index, yearly_trade.values / 1e12, marker='o', linewidth=3, markersize=6, color='red')
plt.title('Trade Value Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Trade Value (Trillion USD)')
plt.grid(True, alpha=0.3)

# 3. Trade by Action (Import/Export)
plt.subplot(2, 3, 3)
action_trade = df.groupby("Action")["Trade Value"].sum()
colors = ['lightcoral', 'lightblue']
wedges, texts, autotexts = plt.pie(action_trade.values, labels=action_trade.index, autopct='%1.1f%%', 
                                  colors=colors, startangle=90)
plt.title('Trade Value by Action', fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_fontweight('bold')

# 4. Trade by Continent
plt.subplot(2, 3, 4)
continent_trade = df.groupby("Continent")["Trade Value"].sum().sort_values(ascending=True)
bars = plt.barh(continent_trade.index, continent_trade.values / 1e12, color='green', alpha=0.7)
plt.title('Trade Value by Continent', fontsize=14, fontweight='bold')
plt.xlabel('Trade Value (Trillion USD)')
for i, v in enumerate(continent_trade.values / 1e12):
    plt.text(v + 0.2, i, f'{v:.1f}T', va='center', fontweight='bold')

# 5. Top 10 Exporters
plt.subplot(2, 3, 5)
exporters = df[df["Action"].str.lower() == "export"].groupby("Country")["Trade Value"].sum().nlargest(10)
bars = plt.bar(range(len(exporters)), exporters.values / 1e12, color='orange', alpha=0.8)
plt.title('Top 10 Exporting Countries', fontsize=14, fontweight='bold')
plt.xlabel('Country')
plt.ylabel('Export Value (Trillion USD)')
plt.xticks(range(len(exporters)), exporters.index, rotation=45, ha='right')
for i, v in enumerate(exporters.values / 1e12):
    plt.text(i, v + 0.05, f'{v:.1f}T', ha='center', va='bottom', fontweight='bold')

# 6. Average Trade Value by Year
plt.subplot(2, 3, 6)
avg_yearly = df.groupby("Year")["Trade Value"].mean()
plt.bar(avg_yearly.index, avg_yearly.values / 1e9, color='purple', alpha=0.7, width=0.8)
plt.title('Average Trade Value per Transaction by Year', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Average Trade Value (Billion USD)')
plt.xticks(rotation=45)

plt.tight_layout(pad=3.0)
plt.savefig('comprehensive_crude_oil_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("=== CRUDE OIL TRADE DATA SUMMARY ===")
print(f"Total records: {len(df):,}")
print(f"Date range: {df['Year'].min()} - {df['Year'].max()}")
print(f"Total countries: {df['Country'].nunique()}")
print(f"Total continents: {df['Continent'].nunique()}")
print(f"Total trade value: ${df['Trade Value'].sum():,.0f}")
print(f"Average trade per transaction: ${df['Trade Value'].mean():,.0f}")
print(f"Highest single transaction: ${df['Trade Value'].max():,.0f}")

# Create a second figure for time series analysis
plt.figure(figsize=(15, 10))

# 1. Yearly trend with percentage change
plt.subplot(2, 2, 1)
yearly_trade = df.groupby("Year")["Trade Value"].sum()
pct_change = yearly_trade.pct_change() * 100
plt.plot(yearly_trade.index, yearly_trade.values / 1e12, marker='o', linewidth=2, label='Trade Value')
plt.title('Yearly Trade Value Trend', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Trade Value (Trillion USD)')
plt.grid(True, alpha=0.3)
plt.legend()

# 2. Year-over-year percentage change
plt.subplot(2, 2, 2)
colors = ['red' if x < 0 else 'green' for x in pct_change.values[1:]]
plt.bar(pct_change.index[1:], pct_change.values[1:], color=colors, alpha=0.7)
plt.title('Year-over-Year Percentage Change', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Percentage Change (%)')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)

# 3. Top countries over time (line plot)
plt.subplot(2, 2, 3)
top_5_countries = df.groupby("Country")["Trade Value"].sum().nlargest(5).index
for country in top_5_countries:
    country_data = df[df["Country"] == country].groupby("Year")["Trade Value"].sum()
    plt.plot(country_data.index, country_data.values / 1e12, marker='o', label=country, linewidth=2)
plt.title('Top 5 Countries Trade Value Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Trade Value (Trillion USD)')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Export vs Import trend
plt.subplot(2, 2, 4)
export_yearly = df[df["Action"].str.lower() == "export"].groupby("Year")["Trade Value"].sum()
import_yearly = df[df["Action"].str.lower() == "import"].groupby("Year")["Trade Value"].sum()
plt.plot(export_yearly.index, export_yearly.values / 1e12, marker='o', label='Exports', linewidth=2, color='blue')
plt.plot(import_yearly.index, import_yearly.values / 1e12, marker='s', label='Imports', linewidth=2, color='red')
plt.title('Exports vs Imports Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Trade Value (Trillion USD)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== VISUALIZATION COMPLETE ===")
print("Generated files:")
print("1. comprehensive_crude_oil_analysis.png - Overview charts")
print("2. time_series_analysis.png - Time series analysis")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("Crude Oil Trade Data Dashboard")

# Load the cleaned data
df = pd.read_csv("C:/Users/clair/Downloads/Global Crude Petroleum Trade 1995-2021.cleaned.csv")

# Sidebar for filtering
st.sidebar.header("Filter Options")
selected_years = st.sidebar.slider("Select Year Range", 
                                  int(df['Year'].min()), 
                                  int(df['Year'].max()), 
                                  (int(df['Year'].min()), int(df['Year'].max())))

selected_continents = st.sidebar.multiselect("Select Continents", 
                                            df['Continent'].unique(), 
                                            default=df['Continent'].unique())

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
col1, col2 = st.columns(2)
with col1:
    action_trade = filtered_df.groupby("Action")["Trade Value"].sum()
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    colors = ['lightcoral', 'lightblue']
    wedges, texts, autotexts = ax3.pie(action_trade.values, labels=action_trade.index, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    ax3.set_title('Trade Value by Action', fontsize=16, fontweight='bold')
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    st.pyplot(fig3)

with col2:
    # Show the actual values
    st.write("**Trade Values by Action:**")
    for action, value in action_trade.items():
        st.write(f"• {action}: ${value/1e12:.1f} Trillion")

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

# 5. Top Exporters vs Importers
st.subheader("Top 10 Exporters vs Importers")
col1, col2 = st.columns(2)

with col1:
    st.write("**Top 10 Exporters**")
    exporters = filtered_df[filtered_df["Action"].str.lower() == "export"].groupby("Country")["Trade Value"].sum().nlargest(10)
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    bars = ax5.bar(range(len(exporters)), exporters.values / 1e12, color='orange', alpha=0.8)
    ax5.set_title('Top 10 Exporting Countries', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Country')
    ax5.set_ylabel('Export Value (Trillion USD)')
    ax5.set_xticks(range(len(exporters)))
    ax5.set_xticklabels(exporters.index, rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig5)

with col2:
    st.write("**Top 10 Importers**")
    importers = filtered_df[filtered_df["Action"].str.lower() == "import"].groupby("Country")["Trade Value"].sum().nlargest(10)
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    bars = ax6.bar(range(len(importers)), importers.values / 1e12, color='blue', alpha=0.8)
    ax6.set_title('Top 10 Importing Countries', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Country')
    ax6.set_ylabel('Import Value (Trillion USD)')
    ax6.set_xticks(range(len(importers)))
    ax6.set_xticklabels(importers.index, rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig6)

# 6. Year-over-year percentage change
st.subheader("Year-over-Year Percentage Change")
yearly_trade = filtered_df.groupby("Year")["Trade Value"].sum()
pct_change = yearly_trade.pct_change() * 100
fig7, ax7 = plt.subplots(figsize=(12, 6))
colors = ['red' if x < 0 else 'green' for x in pct_change.values[1:]]
bars = ax7.bar(pct_change.index[1:], pct_change.values[1:], color=colors, alpha=0.7)
ax7.set_title('Year-over-Year Percentage Change in Trade Value', fontsize=16, fontweight='bold')
ax7.set_xlabel('Year')
ax7.set_ylabel('Percentage Change (%)')
ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax7.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig7)

# Display raw data option
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.dataframe(filtered_df)

# Download option
st.subheader("Download Filtered Data")
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name='filtered_crude_oil_data.csv',
    mime='text/csv'
)
