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