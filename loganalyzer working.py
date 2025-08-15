import os
import duckdb
import streamlit as st
import time
import numpy as np
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"
DUCKDB_PATH = "vectorstore/vector_metadata.duckdb"
DUCKDB_TABLE = "vector_chunks"
CACHE_TTL = 3600

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource(show_spinner=False)
def load_vectorstore(_embedding_model):
    if not os.path.exists(DB_FAISS_PATH):
        st.error(f"Vector store not found at {DB_FAISS_PATH}")
        return None
    db = FAISS.load_local(DB_FAISS_PATH, _embedding_model, allow_dangerous_deserialization=True)
    return db

@st.cache_resource(show_spinner=True)
def load_duckdb_conn():
    if not os.path.exists(DUCKDB_PATH):
        st.error(f"DuckDB database not found at {DUCKDB_PATH}")
        return None
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    return con

def available_fields(con, table):
    result = con.execute(f"PRAGMA table_info({table})").fetchdf()
    return list(result['name'])

def prune_caches(now, ttl):
    """Remove expired entries from both caches."""
    if 'answer_cache' in st.session_state:
        st.session_state.answer_cache = {
            k: v for k, v in st.session_state.answer_cache.items()
            if now - v.get("time", 0) < ttl
        }
    if 'semantic_cache' in st.session_state:
        st.session_state.semantic_cache = [
            entry for entry in st.session_state.semantic_cache
            if now - entry.get("time", 0) < ttl
        ]

def set_custom_prompt():
    prompt_template = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know; don't try to make up an answer.
        Do not provide anything out of the given context.
        If the context is not enough to answer the question, just say that you don't know.
        If the user's question asked for a specific piece of information, try to provide that specific piece of information.
        If the user's question is not clear, ask for clarification.
        If user asks for source IPs, provide the source IPs from the context of the attack attempts.
        If user asks for successful attack attempts, provide the successful attack attempts from the context by checking if the success was TRUE.
        If user asks for failed attack attempts, provide the failed attack attempts from the context by checking if the success was FALSE.
        If user asks for commands used by attackers, provide the commands used by attackers from the context while giving priority to the more complex commands.
        Context: {context}
        Question: {question}
        Let's think step by step.
    """
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])


st.title("Unified Log RAG & SQL Analyzer")

# --- Sidebar navigation instead of tabs (always visible) ---
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("Select your mode:", ["RAG Q&A (Semantic)", "Analytics (SQL Table)"])

with st.sidebar:
    st.markdown("## Tools/Options")
    if st.button("Refresh Vector Store and Embedding Model"):
        st.cache_resource.clear()
        st.success("Vector store and embedding model caches cleared. They will reload on next question.")

    # Separate clear buttons
    if st.button("Clear Chat History"):
        st.session_state['messages'] = []
        st.success("Chat history cleared.")

    if st.button("Clear Answer Cache"):
        st.session_state['answer_cache'] = {}
        st.session_state['semantic_cache'] = []
        st.success("Answer caches cleared.")


# --- Page content logic depending on sidebar selection ---
if page == "RAG Q&A (Semantic)":
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'semantic_cache' not in st.session_state:
        st.session_state.semantic_cache = []  # Each item: {'embedding', 'prompt', 'result', 'source_docs'}
    if 'answer_cache' not in st.session_state:
        st.session_state.answer_cache = {}
    embedding_model = load_embedding_model()
    vectorstore = load_vectorstore(embedding_model)
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

    prompt = st.chat_input("Ask your log question here...")  # --- INPUT IS LAST ---
    if 'last_submitted' not in st.session_state:
        st.session_state['last_submitted'] = ""

    if prompt and vectorstore is not None:
        now = time.time()
        prune_caches(now, CACHE_TTL)

        # Only handle prompt if not just handled (avoids double answer after rerun)
        if prompt != st.session_state['last_submitted']:
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            cache_key = prompt.strip().lower()
            start_time = time.perf_counter()
            from_cache = None

            # --- Cache Checks ---
            cached = st.session_state.answer_cache.get(cache_key)
            if cached and (now - cached.get("time", 0) < CACHE_TTL):
                result = cached["result"]
                source_docs = cached["source_documents"]
                cache_source = "exact"
            else:
                # Semantic cache
                emb_model = embedding_model
                prompt_emb = np.array(emb_model.embed_documents([prompt])[0])
                threshold = 0.93
                semantic_hit = None
                similarity = 0
                for entry in st.session_state.semantic_cache:
                    if now - entry.get("time", 0) > CACHE_TTL:
                        continue
                    emb = entry['embedding']
                    sim = float(np.dot(prompt_emb, emb) / (np.linalg.norm(prompt_emb) * np.linalg.norm(emb)))
                    if sim > threshold:
                        semantic_hit = entry
                        similarity = sim
                        break
                if semantic_hit is not None:
                    result = semantic_hit["result"]
                    source_docs = semantic_hit.get("source_documents", [])
                    cache_source = f"semantic ({similarity:.2f})"
                else:
                    with st.spinner("Generating answer..."):
                        try:
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
                                chain_type_kwargs={'prompt': set_custom_prompt()}
                            )
                            response = qa_chain.invoke({'query': prompt})
                            result = response.get("result", "").strip()
                            source_docs = response.get("source_documents", [])
                        except Exception as e:
                            result = f":red[An error occurred while retrieving your answer: {e}]"
                            source_docs = []
                        
                    # Save to caches only if not error
                    if not result.strip().startswith(":red[An error"):
                        st.session_state.answer_cache[cache_key] = {
                            "result": result,
                            "source_documents": source_docs,
                            "time": now
                        }
                        st.session_state.semantic_cache.append({
                            "prompt": prompt,
                            "embedding": prompt_emb,
                            "result": result,
                            "source_documents": source_docs,
                            "time": now
                        })
                    cache_source = None
    
            end_time = time.perf_counter()
            response_time = end_time - start_time

            timing_info = f"\n\n---\nResponse time: {response_time:.2f}s"
            if cache_source:
                timing_info += f" [cache: {cache_source}]"
            else:
                timing_info += " (live answer)"

            st.session_state.messages.append({
                'role': 'assistant',
                'content': result + timing_info,
                'source_documents': source_docs
            })
            st.session_state['last_submitted'] = prompt
            st.rerun()
        elif vectorstore is None:
            st.warning(":red[Vectorstore failed to load. Please refresh vector store or check your setup.]")


elif page == "Analytics (SQL Table)":
    duck_con = load_duckdb_conn()
    table = DUCKDB_TABLE
    if duck_con is None:
        st.warning(":red[DuckDB not loaded. Please check your database path and refresh.]")
    else:
        if duck_con is not None:
            fields = available_fields(duck_con, table)
            placeholder = "-- Select field --"
            st.header("SQL-powered Metadata Table Explorer")
            mode = st.selectbox("Choose Action", ["Sample", "Count by Field", "Substring Search", "Exact Match"])
            if mode == "Sample":
                st.dataframe(duck_con.execute(f"SELECT * FROM {table} LIMIT 10").fetchdf())
            elif mode == "Count by Field":
                field = st.selectbox("Field", [placeholder]+fields, key="sql_count")
                if field != placeholder:
                    q = f"SELECT {field}, COUNT(*) cnt FROM {table} GROUP BY {field} ORDER BY cnt DESC"
                    st.dataframe(duck_con.execute(q).fetchdf())
            elif mode == "Substring Search":
                field = st.selectbox("Field", [placeholder]+fields, key="sql_substr")
                substr = st.text_input("Substring (case-insensitive)")
                if field != placeholder and substr:
                    q = f"SELECT * FROM {table} WHERE {field} ILIKE '%{substr}%' LIMIT 50"
                    st.dataframe(duck_con.execute(q).fetchdf())
            elif mode == "Exact Match":
                field = st.selectbox("Field", [placeholder]+fields, key="sql_exact")
                value = st.text_input("Value to match exactly")
                if field != placeholder and value:
                    q = f"SELECT * FROM {table} WHERE {field} = ? LIMIT 50"
                    st.dataframe(duck_con.execute(q, [value]).fetchdf())
            if "anomaly_flag" in fields:
                with st.expander("Anomaly Flag Count"):
                    st.dataframe(duck_con.execute(f"SELECT anomaly_flag, COUNT(*) cnt FROM {table} GROUP BY anomaly_flag").fetchdf())
            if "mitre_ttp" in fields:
                with st.expander("Unique MITRE TTPs (first 10)"):
                    st.dataframe(duck_con.execute(f"SELECT DISTINCT mitre_ttp FROM {table} LIMIT 10").fetchdf())
        else:
            st.warning("Load or generate a vector_metadata.duckdb for analytics.")

