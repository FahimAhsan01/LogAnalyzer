import os
import time
import streamlit as st
import numpy as np
from typing import Optional

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"
CACHE_TTL_SECONDS = 3600  # Cache time-to-live in seconds (1 hour)

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

def set_custom_prompt():
    custom_prompt_template = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know; don't try to make up an answer.
        Do not provide anything out of the given context.
        If the context is not enough to answer the question, just say that you don't know.
        If the user's question asked for a specific piece of information, try to provide that specific piece of information.
        If the user's question is not clear, ask for clarification.
        If the user's question requires a specific format, try to provide that format.
        If user asks for source IPs, provide the source IPs from the context of the attack attempts.
        If user asks for successful attack attempts, provide the successful attack attempts from the context by checking if the success was TRUE.

        Context: {context}
        Question: {question}
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables")
        return None
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.4,
        max_new_tokens=512,
        task="conversational",
        token=groq_api_key # type: ignore
    )
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def get_semantic_cached_answer(prompt: str, embedding: np.ndarray, cache: dict, threshold: float = 0.85) -> Optional[tuple]:
    """
    Find cached answer semantically similar to prompt embedding above threshold.
    Returns cached (result, source_docs) or None if no good match.
    """
    now = time.time()
    best_match = None
    best_score = -1.0

    keys_to_delete = []
    for key, entry in cache.items():
        cached_time = entry['timestamp']
        if now - cached_time > CACHE_TTL_SECONDS:
            keys_to_delete.append(key)
            continue
        sim = cosine_similarity(embedding, entry['embedding'])
        if sim > best_score and sim >= threshold:
            best_score = sim
            best_match = entry

    # Evict expired cache entries
    for key in keys_to_delete:
        del cache[key]

    if best_match:
        return best_match['result'], best_match['source_documents']
    return None

def main():
    st.title("The Log Analyzer")

    # Initialize session state entries
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'vectorstore' not in st.session_state:
        embedding_model = load_embedding_model()
        st.session_state.vectorstore = load_vectorstore(embedding_model)
        st.session_state.embedding_model = embedding_model
    if 'response_time' not in st.session_state:
        st.session_state.response_time = None
    if 'answer_cache' not in st.session_state:
        # Cache structure: {key: {embedding: np.array, result: str, source_documents: list, timestamp: float}}
        st.session_state.answer_cache = {}

    # Sidebar controls for clearing history and refreshing vectorstore
    with st.sidebar:
        if st.button("Clear Chat History and Cache"):
            st.session_state.messages = []
            st.session_state.answer_cache = {}
            st.success("Chat history and cache cleared.")
        if st.button("Refresh Vector Store"):
            embedding_model = load_embedding_model()
            st.session_state.vectorstore = load_vectorstore(embedding_model)
            st.session_state.embedding_model = embedding_model
            st.success("Vector store refreshed.")

    # Display chat messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Input prompt
    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        if st.session_state.vectorstore is None:
            st.error("Vector store is not loaded. Please refresh vector store or check logs.")
            return

        # Compute embedding for prompt for semantic cache lookup
        embedding_vec = st.session_state.embedding_model.embed_documents([prompt])
        if embedding_vec:
            embedding_vec = np.array(embedding_vec[0])
        else:
            embedding_vec = None

        cached_answer = None
        if embedding_vec is not None:
            cached_answer = get_semantic_cached_answer(prompt, embedding_vec, st.session_state.answer_cache, threshold=0.95)

        if cached_answer:
            cached_result, cached_source_docs = cached_answer
            st.chat_message('assistant').markdown(cached_result)
            st.session_state.messages.append({'role': 'assistant', 'content': cached_result})
            st.markdown("**Response retrieved from semantic cache**")
            if cached_source_docs:
                st.markdown("---")
                st.markdown("#### Source Documents (Context)")
                with st.expander("Source Documents (Context)", expanded=False):
                    for i, doc in enumerate(cached_source_docs, 1):
                        with st.expander(f"Document {i} metadata: {doc.metadata.get('session', 'N/A')}"):
                            st.write(doc.page_content)
        else:
            # Not in cache, query LLM retrieval pipeline
            try:
                start_time = time.time()

                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatGroq(
                        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # type: ignore
                        temperature=0.5,
                        groq_api_key=os.getenv("GROQ_API_KEY"),  # type: ignore
                        verbose=True
                    ),  # type: ignore
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={'k': 25}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt()}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response.get("result", "").strip()
                source_docs = response.get("source_documents", [])

                # Expanded fallback trigger condition with substring matching
                fallback_triggers = [
                    "no answer returned", "i don't know", "i do not know",
                    "cannot answer", "unknown", "no relevant info", "no information available","the context does not provide information"
                ]
                retrieval_answer_lower = result.lower()

                if not result or any(trigger in retrieval_answer_lower for trigger in fallback_triggers):
                    fallback_prompt = (
                        f"Answer the following question based only on your general knowledge. "
                        f"If you don't know the answer, say so.\nQuestion: {prompt}"
                    )

                    fallback_llm = ChatGroq(
                        model_name="meta-llama/llama-4-maverick-17b-128e-instruct", # type: ignore
                        temperature=0.7,
                        groq_api_key=os.getenv("GROQ_API_KEY"), # type: ignore
                        verbose=True
                    )
                    fallback_response = fallback_llm.invoke(fallback_prompt)

                    fallback_answer = ""
                    if isinstance(fallback_response, dict):
                        fallback_answer = fallback_response.get("text", "").strip()
                    elif isinstance(fallback_response, str):
                        fallback_answer = fallback_response.strip()
                    elif hasattr(fallback_response, "content"):
                        # If fallback_response is an AIMessage or similar, get content attribute
                        fallback_answer = fallback_response.content

                    # st.write("Fallback prompt sent:", fallback_prompt)  # Debug info
                    # st.write("Fallback response received:", repr(fallback_answer))  # Debug info

                    if fallback_answer:
                        st.markdown("**Note:** Provided answer generated by fallback pipeline (LLM only, no retrieval context).")
                        result = fallback_answer
                        source_docs = []

                end_time = time.time()
                latency = end_time - start_time
                st.session_state.response_time = latency

                # Cache the response with timestamp and embedding
                if embedding_vec is not None:
                    st.session_state.answer_cache[prompt] = {
                        'embedding': embedding_vec,
                        'result': result,
                        'source_documents': source_docs,
                        'timestamp': time.time()
                    }

                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})

                st.markdown(f"**Response time:** {latency:.2f} seconds")

                if source_docs:
                    st.markdown("---")
                    st.markdown("#### Source Documents (Context)")
                    with st.expander("Source Documents (Context)", expanded=False):
                        for i, doc in enumerate(source_docs, 1):
                            with st.expander(f"Document {i} metadata: {doc.metadata.get('session', 'N/A')}"):
                                st.write(doc.page_content)

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
