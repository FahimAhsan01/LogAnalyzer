import pyarrow.parquet as pq

# Open your Parquet file
parquet_file = pq.ParquetFile(r"processed_data\cowrie_events_20250814_195008.parquet")

# Access the schema as Arrow schema
schema = parquet_file.schema_arrow

# Get the list of column names
column_names = schema.names

print(column_names)
