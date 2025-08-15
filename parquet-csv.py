import pandas as pd

# Replace with your Parquet file path
file_path = "data/"

# Load Parquet into a DataFrame
df = pd.read_parquet(file_path)

# Inspect the first few rows
print(df.head())

# Get a summary of the columns and data types
print(df.info())

# Save to CSV
df.to_csv("data/", index=False)
