import os
import streamlit as st
import pandas as pd
import duckdb
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from collections import Counter
import threading
import queue

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"
PARQUET_DATA_DIR = "processed_data"

# Thread-safe queues for async processing
query_queue = queue.Queue()
response_queue = queue.Queue()

@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def run_rag_pipeline(query, vectorstore):
    prompt_template = """
        Use the provided context to answer the user's question.
        If answer is unknown, say "I don't know."
        Context: {context}
        Question: {question}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.5,
            groq_api_key=GROQ_API_KEY,
            verbose=False
        ),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 25}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    response = qa_chain.invoke({"query": query})
    return response

def run_sql_pipeline(query):
    numeric_triggers = ["count", "number", "how many", "total", "sum", "average", "max", "min", "frequency", "occurence"]

    if not any(kw in query.lower() for kw in numeric_triggers):
        return None

    con = duckdb.connect()
    parquet_pattern = os.path.join(PARQUET_DATA_DIR, '*.parquet')

    sql_base = f"WITH all_data AS (SELECT * FROM read_parquet('{parquet_pattern}')) "

    if any(x in query.lower() for x in ["count", "how many", "number"]):
        sql = sql_base + "SELECT COUNT(*) AS count FROM all_data"
    else:
        sql = sql_base + query

    try:
        df = con.execute(sql).fetchdf()
        con.close()
        if df.shape[1] == 1:
            return f"Result: {df.iloc[0,0]}"
        else:
            return df
    except Exception as e:
        con.close()
        return f"SQL execution error: {str(e)}"

def run_fallback_pipeline(query):
    llm = ChatGroq(
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.7,
        groq_api_key=GROQ_API_KEY
    )
    response = llm.invoke(query)
    return getattr(response, "generated_text", "Sorry, I could not provide an answer.")

# Background worker thread processing queries asynchronously
def worker_thread_func(vectorstore):
    numeric_triggers = ["count", "number", "how many", "total", "sum", "average", "max", "min", "frequency", "occurence"]

    while True:
        query = query_queue.get()
        if query == "__quit__":
            break

        # Step 1: Run RAG pipeline
        rag_response = run_rag_pipeline(query, vectorstore)
        rag_answer = rag_response.get("result", "")
        rag_sources = rag_response.get("source_documents", [])

        if rag_answer and "don't know" not in rag_answer.lower():
            answer = rag_answer
            pipeline_used = "RAG"
        else:
            # Step 2: Run SQL pipeline only if query is numeric-related
            if any(kw in query.lower() for kw in numeric_triggers):
                sql_answer = run_sql_pipeline(query)
                if sql_answer is not None:
                    answer = str(sql_answer)
                    pipeline_used = "SQL"
                else:
                    # Step 3: Fallback pipeline if both fail
                    answer = run_fallback_pipeline(query)
                    pipeline_used = "Fallback"
            else:
                # Non-numeric query falls back directly
                answer = run_fallback_pipeline(query)
                pipeline_used = "Fallback"

        response_queue.put({
            "query": query,
            "answer": answer,
            "pipeline_used": pipeline_used,
            "rag_sources": rag_sources
        })

def dashboard_panel(query_history):
    st.header("Analytics Dashboard")

    if not query_history:
        st.write("No queries yet. Start chatting!")
        return

    df = pd.DataFrame(query_history)

    st.subheader("Total Queries")
    st.metric("Number of Queries", len(df))

    st.subheader("Pipeline Usage")
    st.bar_chart(df["pipeline"].value_counts())

    all_words = " ".join(df['query'].astype(str)).lower().split()
    top_words = Counter(all_words).most_common(10)
    word_df = pd.DataFrame(top_words, columns=["Keyword", "Count"])
    st.subheader("Top Keywords")
    st.bar_chart(word_df.set_index("Keyword"))

    src_ips = []
    for sources in df.get('rag_sources', []):
        for doc in sources:
            metadata = doc.metadata
            if isinstance(metadata, dict):
                ip = metadata.get('src_ip')
            else:
                ip = getattr(metadata, 'src_ip', None)
            if ip:
                src_ips.append(ip)
    if src_ips:
        ip_counts = pd.Series(src_ips).value_counts().head(10)
        st.subheader("Top Source IPs")
        st.bar_chart(ip_counts)

    commands = []
    for sources in df.get('rag_sources', []):
        for doc in sources:
            metadata = doc.metadata
            if isinstance(metadata, dict):
                cmd = metadata.get('command')
            else:
                cmd = getattr(metadata, 'command', None)
            if cmd:
                commands.append(cmd)
    if commands:
        cmd_counts = pd.Series(commands).value_counts().head(10)
        st.subheader("Top Commands")
        st.bar_chart(cmd_counts)

def main():
    st.title("Async Multi-Pipeline RAG Chatbot with Dashboard")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    vectorstore = load_vectorstore()

    # Start background worker thread once
    if "worker_thread" not in st.session_state:
        thread = threading.Thread(target=worker_thread_func, args=(vectorstore,), daemon=True)
        thread.start()
        st.session_state.worker_thread = thread

    chat_col, dash_col = st.columns([3, 1])

    with chat_col:
        for msg in st.session_state.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            st.markdown(f"**{role.capitalize()}:** {content}")

        with st.form("input_form", clear_on_submit=True):
            user_input = st.text_input("Ask your question here", disabled=st.session_state.is_processing)
            submitted = st.form_submit_button("Send")

            if submitted and user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.is_processing = True
                query_queue.put(user_input)

    # Poll for response and update UI when available
    if not response_queue.empty():
        res = response_queue.get()
        st.session_state.is_processing = False
        st.session_state.messages.append({"role": "assistant", "content": res["answer"], "pipeline": res["pipeline_used"]})
        st.session_state.query_history.append({
            "query": res["query"],
            "answer": res["answer"],
            "pipeline": res["pipeline_used"],
            "rag_sources": res["rag_sources"] if res["pipeline_used"] == "RAG" else [],
        })
        st.rerun()

    if st.session_state.is_processing:
        st.info("Generating answer... please wait.")

    with dash_col:
        dashboard_panel(st.session_state.query_history)

if __name__ == "__main__":
    main()
