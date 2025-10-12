"""Utilities to load, clean, and save the Global Crude Petroleum Trade 1995-2021 dataset."""

import os
import pandas as pd

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
# df = load_crude_trade(r"C:\Users\clair\Downloads\Global Crude Petroleum Trade 1995-2021.csv")
	df_clean = clean_crude_trade(df)
	print("Cleaned shape:", df_clean.shape)
	# Add ID column starting from 1
	df_clean.insert(0, "ID", range(1, len(df_clean) + 1))
import pandas as pd

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
	africa_df = df_clean[df_clean["Continent"].str.lower() == "africa"]
	print("\nCleaned data for Africa:")
	print(africa_df.to_string(index=False))
	csv_out, parquet_out = save_cleaned(df_clean, out_dir, basename)
	print("Saved cleaned CSV:", csv_out)
	print("Saved cleaned Parquet:", parquet_out)
	
print(df.head(10))  # Shows column names and first 10 rows
print(df.head(10).reset_index(drop=True))

print("\nTable with ID column (first 10 rows):")
