import os
import duckdb
import streamlit as st
import time
import numpy as np
import psutil
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv

import pydeck as pdk
import plotly.express as px
import pandas as pd
import matplotlib

# ------- CONFIGURATION (from env, sidebar, or defaults) -------
load_dotenv(find_dotenv())
DEFAULT_DB_FAISS_PATH = os.getenv("DB_FAISS_PATH", "vectorstore/db_faiss")
DEFAULT_DUCKDB_PATH = os.getenv("DUCKDB_PATH", "vectorstore/vector_metadata.duckdb")
DEFAULT_DUCKDB_TABLE = os.getenv("DUCKDB_TABLE", "vector_chunks")
DEFAULT_CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

# ---- SESSION STATE INIT ----
if 'query_log' not in st.session_state:
    st.session_state['query_log'] = []
if 'error_log' not in st.session_state:
    st.session_state['error_log'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'semantic_cache' not in st.session_state:
    st.session_state['semantic_cache'] = []
if 'answer_cache' not in st.session_state:
    st.session_state['answer_cache'] = {}
# Initialization (once, e.g. at app startup)
    
ORIGINAL_RAG_PROMPT = """
You are a helpful cybersecurity assistant. Use the pieces of information provided in the context below to answer the user's question as clearly, concisely, and specifically as possible.

- Retrieve the most relevant and latest documents for this question focusing on current date/year first.
- Base your answer only on the context—do not add facts or details not present in the context.
- If the answer can be found or reasoned from the context, respond directly and concretely.
- If the answer cannot be determined from the context, simply reply: "I don't know."
- If the question is ambiguous or cannot be answered with the provided information, ask the user to clarify.
- If user asks for specific fields (such as source IPs, failed/successful attempts, usernames, commands), extract and summarize those from the context where possible.
- Prefer clear lists, tables, or bullet points if several facts are relevant.
- Do not repeat the entire context—make your answer focused and relevant to the question.

"""
FIXED_PROMPT_FOOTER = """
Context:
{context}

Question:
{question}

Answer:
"""

if 'rag_instruction' not in st.session_state:
    st.session_state['rag_instruction'] = ORIGINAL_RAG_PROMPT
# ------- CORE LOGIC -------

@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

@st.cache_resource(show_spinner=False)
def load_vectorstore(_embedding_model, path):
    if not os.path.exists(path):
        st.error(f"Vector store not found at {path}")
        return None
    db = FAISS.load_local(path, _embedding_model, allow_dangerous_deserialization=True)
    return db

@st.cache_resource(show_spinner=True)
def load_duckdb_conn(path):
    if not os.path.exists(path):
        st.error(f"DuckDB database not found at {path}")
        return None
    con = duckdb.connect(path, read_only=True)
    return con

def available_fields(con, table):
    result = con.execute(f"PRAGMA table_info({table})").fetchdf()
    return sorted(list(result['name']))

def default_prompt_template():
    return st.session_state['rag_prompt']

def record_query(prompt, latency_s, from_cache):
    st.session_state['query_log'].append({
        "prompt": prompt,
        "latency": latency_s,
        "cache": from_cache,
        "timestamp": time.time()
    })

def dataframe_download_button(df, label="Download as CSV"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label, csv, file_name="results.csv", mime="text/csv")

def display_resource_stats():
    cpu = psutil.cpu_percent(interval=0.2)
    ram = psutil.virtual_memory().percent
    st.markdown(f"**CPU Usage:** {cpu}%")
    st.markdown(f"**RAM Usage:** {ram}%")

def get_vectorstore_stats(vectorstore):
    try:
        return {
            "Vectors": getattr(vectorstore.index, 'ntotal', '?'),
            "Dimension": getattr(vectorstore.index, 'd', '?'),
            "Clusters": getattr(vectorstore.index, 'nlist', '?')
        }
    except Exception:
        return {}

def get_duckdb_stats(duck_con, table):
    try:
        row_count = duck_con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return {"Rows": row_count}
    except Exception as e:
        return {"Rows": f"Error: {e}"}

# ------- SIDEBAR (ALL IN ONE BLOCK) -------
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("Choose your Feature:", ["RAG Q&A (Semantic)", "Analytics (SQL Table)", "Attack Map"])
    st.markdown("## Tools / Options")
    if st.button("Refresh All Caches (Vector, Embedding, DB)"):
        st.cache_resource.clear()
        st.success("Caches cleared, will reload on next query.")
    if st.button("Clear Chat History"):
        st.session_state['messages'] = []
        st.success("Chat history cleared.")
    if st.button("Clear Answer Cache"):
        st.session_state['answer_cache'] = {}
        st.session_state['semantic_cache'] = []
        st.success("Answer caches cleared.")

    # CONFIGURATION
    with st.expander("Configuration (click to expand/collapse)", expanded=False):
        DB_FAISS_PATH = st.text_input("FAISS index path", value=DEFAULT_DB_FAISS_PATH, key="db_faiss")
        DUCKDB_PATH = st.text_input("DuckDB path", value=DEFAULT_DUCKDB_PATH, key="duckdb_path")
        DUCKDB_TABLE = st.text_input("DuckDB Table", value=DEFAULT_DUCKDB_TABLE, key="duckdb_table")
        CACHE_TTL = st.number_input("Cache TTL (s)", min_value=60, max_value=86400, value=DEFAULT_CACHE_TTL, step=60, key="cache_ttl")
        if st.checkbox("Edit RAG Prompt Template (Advanced)", value=False, key="edit_prompt"):
            st.session_state['rag_instruction'] = st.text_area(
                "Edit Instructions", value=st.session_state['rag_instruction'], height=250
            )
            
    full_prompt = st.session_state['rag_instruction'] + FIXED_PROMPT_FOOTER

    # APP MONITORING
    SHOW_ADVANCED = st.checkbox("Show Advanced App Monitoring Panel", value=True, key="show_adv")
    if SHOW_ADVANCED:
        with st.expander("App Monitoring & Stats", expanded=False):
            display_resource_stats()
            if 'vector_stats' in st.session_state:
                st.markdown("**VectorStore Stats:**")
                for k, v in st.session_state['vector_stats'].items():
                    st.write(f"{k}: {v}")
            if 'db_stats' in st.session_state:
                st.markdown("**DuckDB Stats:**")
                for k, v in st.session_state['db_stats'].items():
                    st.write(f"{k}: {v}")
            ql = st.session_state['query_log']
            if ql:
                last_n = st.number_input("Show latency for last N queries", min_value=1, max_value=1000, value=min(100, len(ql)), key="latency_n")
                logs = ql[-last_n:]
                avg_latency = np.mean([q['latency'] for q in logs])
                st.write(f"Avg Latency: {avg_latency:.2f}s")
                cache_hits = sum(1 for q in logs if q['cache'])
                st.write(f"Cache Hits: {cache_hits}/{len(logs)} ({int(cache_hits/len(logs)*100)}%)")
            if st.session_state['error_log']:
                st.markdown("***Errors:***")
                for err in st.session_state['error_log'][-5:]:
                    st.write(err)

# ----- MAIN PAGE LOGIC BELOW -----
if page == "RAG Q&A (Semantic)":
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if not st.session_state['messages']:
        st.session_state['messages'].append({
            'role': 'assistant',
            'content': "Hello! Welcome to the PMICS Log Analyzer Chatbot. Ask me anything about your logs or analytics."
        })
        st.session_state.semantic_cache = []  # Each item: {'embedding', 'prompt', 'result', 'source_docs'}
    if 'answer_cache' not in st.session_state:
        st.session_state.answer_cache = {}
    embedding_model = load_embedding_model()
    vectorstore = load_vectorstore(embedding_model, DB_FAISS_PATH)
    st.header("Semantic Log Q&A")



    # Show chat history (user/assistant, top-to-bottom, input always last)
    for i, message in enumerate(st.session_state.messages):
        st.chat_message(message['role']).markdown(message['content'])
        if message['role'] == 'assistant':
            if i > 0 and 'source_documents' in st.session_state.messages[i-1]:
                resp_docs = st.session_state.messages[i-1]['source_documents']
                if resp_docs:
                    with st.expander("Source Documents (click to expand/collapse all)", expanded=False):
                        for j, doc in enumerate(resp_docs, 1):
                            md = doc.metadata
                            label = f"Source {j}: session={md.get('session','?')}, time={md.get('timestamp','?')}"
                            with st.expander(label, expanded=False):
                                st.markdown(doc.page_content if doc.page_content else "*No page content*")
                                if md:
                                    st.markdown("**Metadata:**")
                                    st.json(md)
    
    qa_prompt_template = PromptTemplate(
        template=full_prompt,
        input_variables=["context", "question"]
    )

    user_query = st.chat_input("Ask your log question here...")

    if user_query and vectorstore is not None:
        st.session_state.messages.append({'role': 'user', 'content': user_query})
        cache_key = user_query.strip().lower()
        start_time = time.perf_counter()
        now = time.time()
        from_cache = None

        # 1. Exact cache
        cached = st.session_state.answer_cache.get(cache_key)
        if cached and (now - cached.get("time", 0) < CACHE_TTL):
            result = cached["result"]
            source_docs = cached["source_documents"]
            from_cache = "exact"
        else:
            # 2. Semantic cache
            emb_model = embedding_model
            user_emb = np.array(emb_model.embed_documents([user_query])[0])
            threshold = 0.93
            semantic_hit = None
            similarity = 0
            for entry in st.session_state.semantic_cache:
                if now - entry.get("time", 0) > CACHE_TTL:
                    continue
                emb = entry['embedding']
                sim = float(np.dot(user_emb, emb) / (np.linalg.norm(user_emb) * np.linalg.norm(emb)))
                if sim > threshold:
                    semantic_hit = entry
                    similarity = sim
                    break

            if semantic_hit is not None:
                result = semantic_hit["result"]
                source_docs = semantic_hit.get("source_documents", [])
                from_cache = f"semantic ({similarity:.2f})"
            else:
                # 3. No cache hit: generate answer
                with st.spinner("Generating answer..."):
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=ChatGroq(
                            model_name="meta-llama/llama-4-maverick-17b-128e-instruct", # type: ignore
                            temperature=0.5,
                            groq_api_key=os.getenv("GROQ_API_KEY"), # type: ignore
                            verbose=True,
                        ),
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={'k': 25}),
                        return_source_documents=True,
                        chain_type_kwargs={'prompt': qa_prompt_template}
                    )
                    response = qa_chain.invoke({'query': user_query})
                    result = response.get("result", "").strip()
                    source_docs = response.get("source_documents", [])

                # Save to both caches for future
                st.session_state.answer_cache[cache_key] = {
                    "result": result,
                    "source_documents": source_docs,
                    "time": now
                }
                st.session_state.semantic_cache.append({
                    "prompt": user_query,
                    "embedding": user_emb,
                    "result": result,
                    "source_documents": source_docs,
                    "time": now
                })

        end_time = time.perf_counter()
        response_time = end_time - start_time

        timing_info = f"\n\n---\nResponse time: {response_time:.2f}s"
        if from_cache:
            timing_info += f" [cache: {from_cache}]"
        else:
            timing_info += " (live answer)"

        record_query(
            prompt=user_query,
            latency_s=response_time,
            from_cache=from_cache
        )

        st.session_state.messages.append({
            'role': 'assistant',
            'content': result + timing_info,
            'source_documents': source_docs
        })
        st.rerun()


elif page == "Analytics (SQL Table)":
    duck_con = load_duckdb_conn(DUCKDB_PATH)
    table = DUCKDB_TABLE
    if duck_con is not None:
        fields = available_fields(duck_con, table)
        placeholder = "-- Select field --"
        st.header("SQL-powered Metadata Table Explorer")
        mode = st.selectbox("Choose Action", ["Sample", "Count by Field", "Substring Search", "Exact Match", "Custom SQL"])
        if mode == "Sample":
            df = duck_con.execute(f"SELECT * FROM {table} LIMIT 10").fetchdf()
            st.dataframe(df)
            dataframe_download_button(df)
        elif mode == "Count by Field":
            field = st.selectbox("Field", [placeholder]+fields, key="sql_count")
            if field != placeholder:
                q = f"SELECT {field}, COUNT(*) cnt FROM {table} GROUP BY {field} ORDER BY cnt DESC"
                df = duck_con.execute(q).fetchdf()
                st.dataframe(df)
                dataframe_download_button(df)
        elif mode == "Substring Search":
            field = st.selectbox("Field", [placeholder]+fields, key="sql_substr")
            substr = st.text_input("Substring (case-insensitive)")
            if field != placeholder and substr:
                q = f"SELECT * FROM {table} WHERE {field} ILIKE '%{substr}%' LIMIT 50"
                df = duck_con.execute(q).fetchdf()
                st.dataframe(df)
                dataframe_download_button(df)
        elif mode == "Exact Match":
            field = st.selectbox("Field", [placeholder]+fields, key="sql_exact")
            value = st.text_input("Value to match exactly")
            if field != placeholder and value:
                q = f"SELECT * FROM {table} WHERE {field} = ? LIMIT 50"
                df = duck_con.execute(q, [value]).fetchdf()
                st.dataframe(df)
                dataframe_download_button(df)
        elif mode == "Custom SQL":
            sql_query = st.text_area("Enter a custom SQL query for DuckDB", height=80)
            if st.button("Execute Query"):
                try:
                    df = duck_con.execute(sql_query).fetchdf()
                    st.dataframe(df)
                    dataframe_download_button(df)
                except Exception as e:
                    st.error(f"SQL Error: {e}")

        if "anomaly_flag" in fields:
            with st.expander("Anomaly Flag Count"):
                st.dataframe(duck_con.execute(f"SELECT anomaly_flag, COUNT(*) cnt FROM {table} GROUP BY anomaly_flag").fetchdf())
        if "mitre_ttp" in fields:
            with st.expander("Unique MITRE TTPs (first 10)"):
                st.dataframe(duck_con.execute(f"SELECT DISTINCT mitre_ttp FROM {table} LIMIT 10").fetchdf())
    else:
        st.warning("Load or generate a vector_metadata.duckdb for analytics.")


elif page == "Attack Map":
    st.header("Geospatial visualization of attack source IPs")

    duck_con = load_duckdb_conn(DUCKDB_PATH)
    table = DUCKDB_TABLE

    if duck_con is not None:
        try:
            df = duck_con.execute(
                f"""
                SELECT
                    src_ip,
                    CAST(split_part(location, ',', 1) AS DOUBLE) AS lat,
                    CAST(split_part(location, ',', 2) AS DOUBLE) AS lon,
                    country,
                    region,
                    city,
                    org,
                    asn,
                    timestamp
                FROM {table}
                WHERE location IS NOT NULL
                LIMIT 1000
                """
            ).fetchdf()
        except Exception as e:
            st.error(f"Could not get location data: {e}")
            df = None

        if df is not None and not df.empty:
            # Filter out bad coordinates
            df = df[(df['lat'] != 0) & (df['lon'] != 0)]

            # --- 1. Pydeck Interactive Map ---

            tooltip = {
                "html": (
                    "<b>Source IP:</b> {src_ip}<br>"
                    "<b>Country:</b> {country}<br>"
                    "<b>Region:</b> {region}<br>"
                    "<b>City:</b> {city}<br>"
                    "<b>Org:</b> {org}<br>"
                    "<b>ASN:</b> {asn}<br>"
                    "<b>Time:</b> {timestamp}"
                ),
                "style": {"backgroundColor": "bluesteel", "color": "white"},
            }

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position='[lon, lat]',
                get_radius=40000,
                get_fill_color='[200, 30, 0, 160]',
                pickable=True,
                auto_highlight=True,
            )

            pdk_map = pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(
                    latitude=df['lat'].mean() if not df['lat'].isna().all() else 0,
                    longitude=df['lon'].mean() if not df['lon'].isna().all() else 0,
                    zoom=3,
                    pitch=15,
                ),
                map_provider="carto",
                map_style='dark',
            )
            st.pydeck_chart(pdk_map)

            # --- 2. Plotly Bar: Top Countries ---
            top_countries = df['country'].value_counts().head(10).reset_index()
            top_countries.columns = ['country', 'attack_count']
            fig_countries = px.bar(
                top_countries,
                x='country', y='attack_count',
                color='attack_count', color_continuous_scale='reds',
                title="Top Countries by Attack Source",
                labels={'attack_count':'Number of Attacks'}
            )
            fig_countries.update_layout(template="plotly_dark")
            st.plotly_chart(fig_countries, use_container_width=True)

            # --- 3. Plotly Bar: Top Organizations/ASNs ---
            top_orgs = df['org'].fillna(df['asn']).value_counts().head(10).reset_index()
            top_orgs.columns = ['org', 'attack_count']
            fig_orgs = px.bar(
                top_orgs,
                x='org', y='attack_count',
                color='attack_count', color_continuous_scale='blues',
                title="Top Organizations by Attack Source",
                labels={'attack_count':'Number of Attacks'}
            )
            fig_orgs.update_layout(template="plotly_dark")
            st.plotly_chart(fig_orgs, use_container_width=True)

            # --- 4. Optional Pie Chart: Top Cities ---
            top_cities = df['city'].value_counts().head(8).reset_index()
            top_cities.columns = ['city', 'attack_count']
            fig_cities = px.pie(
                top_cities,
                names='city', values='attack_count',
                title="Attack Distribution by City"
            )
            fig_cities.update_layout(template="plotly_dark")
            st.plotly_chart(fig_cities, use_container_width=True)

            # --- 5. Styled Data Table ---
            with st.expander("Detailed attack data", expanded=False):
                st.dataframe(
                    df[["src_ip", "country", "region", "city", "org", "asn", "timestamp", "lat", "lon"]].style.background_gradient(cmap="OrRd"),
                    use_container_width=True
                )

        else:
            st.info("No enriched points found for plotting. Try a different dataset, or check your 'location' column.")

    else:
        st.warning("Could not load attack data from DuckDB. Please check your configuration or data file.")
        
