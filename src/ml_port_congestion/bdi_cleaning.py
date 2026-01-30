
import pandas as pd
import os

# Get the project root directory (assumes script is run from anywhere)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_PATH = os.path.join(DATA_DIR, 'bdi_raw.csv')
CLEAN_PATH = os.path.join(DATA_DIR, 'bdi_clean.csv')

# Load the raw file
df = pd.read_csv(RAW_PATH)

print("Original Columns:", df.columns)

# Selecting important datas
df = df.iloc[:, 0:2] # Keep only first 2 columns
df.columns = ['Date', 'BDI'] # Rename column names
    
# Date formating: mm/DD/YYYY format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Remove quotes and commas
def clean_currency(x):
    if isinstance(x, str):
        return float(x.replace(',', '').replace('"', ''))
    return float(x)

df['BDI'] = df['BDI'].apply(clean_currency)

# 4. Sort Oldest to Newest
# Machine Learning needs time to flow forward (2019 -> 2025)
df = df.sort_values('Date').reset_index(drop=True)

# 5. Save
df.to_csv(CLEAN_PATH, index=False)