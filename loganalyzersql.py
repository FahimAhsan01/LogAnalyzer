import streamlit as st
import duckdb
import pandas as pd
import json
import os
import numpy as np

DEFAULT_TABLE = "cowrie_events"

def convert_ndarray(obj):
    """Recursively convert any np.ndarray in lists/dicts to native Python lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarray(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    else:
        return obj

def flatten_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            if df[col].apply(lambda x: isinstance(x, (dict, list, np.ndarray))).any():
                df[col] = df[col].apply(lambda x:
                    json.dumps(convert_ndarray(x)) if isinstance(x, (dict, list, np.ndarray)) else x
                )
    return df


def load_parsed_to_duckdb(df: pd.DataFrame, db_path: str = ":memory:", table: str = DEFAULT_TABLE) -> duckdb.DuckDBPyConnection:
    df = flatten_for_sql(df)
    con = duckdb.connect(database=db_path)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df")
    return con

def available_fields(con: duckdb.DuckDBPyConnection, table: str = DEFAULT_TABLE):
    result = con.execute(f"PRAGMA table_info({table})").fetchdf()
    return list(result['name'])

def count_by_field(con: duckdb.DuckDBPyConnection, field: str, where: str = "", table: str = DEFAULT_TABLE):
    query = f"SELECT {field}, COUNT(*) as cnt FROM {table} {where} GROUP BY {field} ORDER BY cnt DESC"
    return con.execute(query).fetchdf()

def unique_values(con: duckdb.DuckDBPyConnection, field: str, table: str = DEFAULT_TABLE, limit: int = 100):
    query = f"SELECT DISTINCT {field} FROM {table} LIMIT {limit}"
    return con.execute(query).fetchdf()

def match_and_show(con: duckdb.DuckDBPyConnection, field: str, value, table: str = DEFAULT_TABLE, limit: int = 100):
    query = f"SELECT * FROM {table} WHERE {field} = ? LIMIT {limit}"
    return con.execute(query, [value]).fetchdf()

def search_substring(con: duckdb.DuckDBPyConnection, field: str, substr: str, table: str = DEFAULT_TABLE, limit: int = 100):
    query = f"SELECT * FROM {table} WHERE CAST({field} AS VARCHAR) ILIKE ? LIMIT {limit}"
    return con.execute(query, [f"%{substr}%"]).fetchdf()

def count_anomalies(con: duckdb.DuckDBPyConnection, table: str = DEFAULT_TABLE):
    return con.execute(f"SELECT anomaly_flag, COUNT(*) as cnt FROM {table} GROUP BY anomaly_flag").fetchdf()

def sample_query(con: duckdb.DuckDBPyConnection, limit=10, table: str = DEFAULT_TABLE):
    return con.execute(f"SELECT * FROM {table} LIMIT {limit}").fetchdf()

#### STREAMLIT UI ####

st.title("Cowrie SQL Event Analyzer (Streamlit)")

# ------- Data Load Panel --------
st.sidebar.header("Step 1: Upload/Select Data File")
data_path = st.sidebar.text_input("Path to processed log (.parquet/.csv/.jsonl):", value="")
db_path = st.sidebar.text_input("(Optional) DuckDB file on disk:", value=":memory:")

if st.sidebar.button("Load Data"):
    if not data_path or not os.path.isfile(data_path):
        st.error("Please supply a valid path to a processed file.")
        st.stop()
    with st.spinner("Loading data..."):
        if data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        elif data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        elif data_path.endswith(".jsonl") or data_path.endswith(".json"):
            df = pd.read_json(data_path, lines=True)
        else:
            st.error("Unknown file extension.")
            st.stop()
        st.session_state.con = load_parsed_to_duckdb(df, db_path=db_path)
        st.session_state.df = df
        st.success(f"Loaded {len(df)} rows - {len(df.columns)} fields")

if "con" in st.session_state:
    con = st.session_state.con
    df = st.session_state.df
else:
    st.info("Upload and load a dataset to begin.")
    st.stop()

fields = available_fields(con)
placeholder_option = "-- Select field --"

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Sample", "Count by Field", "Unique Values", "Search (Exact)", "Search (Substring)"]
)

with tab1:
    st.header("First 10 Event Rows")
    st.dataframe(sample_query(con, limit=10))

with tab2:
    st.header("Count/Group Events by Field")
    field = st.selectbox("Select field/column", [placeholder_option] + fields, key="count_field")
    if st.button("Count!", key="count_btn"):
        if field == placeholder_option or not field:
            st.warning("Please select a field before counting.")
        else:
            st.dataframe(count_by_field(con, field))

with tab3:
    st.header("Show Unique Values")
    field = st.selectbox("Field for unique values", [placeholder_option] + fields, key="uniq_field")
    limit = st.slider("Limit", 5, 100, 20)
    if st.button("Show Unique", key="uniq_btn"):
        if field == placeholder_option or not field:
            st.warning("Please select a field for unique values.")
        else:
            st.dataframe(unique_values(con, field, limit=limit))

with tab4:
    st.header("Search for Exact Matches")
    field = st.selectbox("Field to search", [placeholder_option] + fields, key="exact_field")
    value = st.text_input("Match this value exactly:")
    limit = st.slider("Show at most", 5, 100, 20, key="exact_limit")
    if st.button("Show Matches", key="exact_btn"):
        if field == placeholder_option or not field:
            st.warning("Please select a field for searching.")
        elif value == "":
            st.warning("Please enter a value to match exactly.")
        else:
            st.dataframe(match_and_show(con, field, value, limit=limit))

with tab5:
    st.header("Search for Substring")
    field = st.selectbox("Field to scan", [placeholder_option] + fields, key="substr_field")
    substr = st.text_input("Substring to search for (case-insensitive):")
    limit = st.slider("Max rows", 5, 100, 20, key="substr_limit")
    if st.button("Search Substring", key="substr_btn"):
        if field == placeholder_option or not field:
            st.warning("Please select a field for substring search.")
        elif substr == "":
            st.warning("Please enter a substring to search.")
        else:
            st.dataframe(search_substring(con, field, substr, limit=limit))

# ---- Anomaly/MITRE quick-view -----
if "anomaly_flag" in fields:
    with st.expander("Anomaly Flag Count"):
        st.dataframe(count_anomalies(con))
if "mitre_ttp" in fields:
    with st.expander("Unique MITRE TTPs (first 10)"):
        st.dataframe(unique_values(con, "mitre_ttp", limit=10))
