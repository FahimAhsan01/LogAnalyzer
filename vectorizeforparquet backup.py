import os
import json
from typing import List
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ==== Configuration ====
DATA_DIR = "processed_data"
VECTOR_DB_PATH = Path("vectorstore/db_faiss")
PROCESSED_LOG_FILE = "processed_files_log.json"  # Tracks processed Parquet files with timestamps

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
EMBEDDING_BATCH_SIZE = 512
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

REQUIRED_FIELDS = ["timestamp", "src_ip", "session", "command", "mitre_ttp"]

def load_processed_files_log(log_path: str) -> dict:
    """Load processed files log which stores filenames and last processed modification times."""
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return json.load(f)
    return {}

def save_processed_files_log(log_path: str, data: dict):
    """Save the processed files log."""
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)

def get_new_parquet_files(directory: str, processed_files: dict) -> List[str]:
    """Return list of unprocessed or modified Parquet files."""
    all_files = [f for f in os.listdir(directory) if f.endswith(".parquet")]
    new_files = []

    for file in all_files:
        file_path = os.path.join(directory, file)
        mod_time = os.path.getmtime(file_path)

        if file not in processed_files or mod_time > processed_files[file]:
            new_files.append(file)
    return new_files

def validate_required_fields(df: pd.DataFrame, required_fields: List[str]) -> bool:
    missing = [field for field in required_fields if field not in df.columns]
    if missing:
        print(f"[Warning] Missing required fields {missing} in batch. Skipping batch.")
        return False
    return True

def load_parquet_documents(file_paths: List[str], directory: str) -> List[Document]:
    documents = []

    for filename in tqdm(file_paths, desc="Loading new Parquet files"):
        file_path = os.path.join(directory, filename)
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=1000):
                df = batch.to_pandas()

                if not validate_required_fields(df, REQUIRED_FIELDS):
                    continue

                # Normalize datetime columns
                datetime_cols = [col for col in df.columns if 'time' in col.lower()]
                for col in datetime_cols:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

                for _, row in df.iterrows():
                    if row is None or row.empty:
                        continue

                    content_parts = []
                    for col, value in row.items():
                        if isinstance(value, (list, np.ndarray)):
                            content_parts.append(f"{col}: {', '.join(map(str, value))}")
                        elif pd.api.types.is_scalar(value) and pd.notna(value):
                            content_parts.append(f"{col}: {value}")
                    page_content = " | ".join(content_parts)

                    metadata = row.to_dict()
                    for k, v in metadata.items():
                        if isinstance(v, (np.generic, np.ndarray)):
                            metadata[k] = v.item() if np.ndim(v) == 0 else v.tolist()

                    documents.append(Document(page_content=page_content, metadata=metadata))

        except ImportError:
            print("Error: pyarrow is required for Parquet processing. Please install it.")
            return []
        except Exception as e:
            print(f"[Error] Failed processing {filename}: {str(e)}")

    return documents

def create_vector_db(documents: List[Document]):
    if not documents:
        print("No documents to process for vector database creation.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    text_chunks = splitter.split_documents(documents)
    print(f"Creating vector database from {len(text_chunks)} chunks (from {len(documents)} documents)")

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    all_embeddings = []

    for i in tqdm(range(0, len(text_chunks), EMBEDDING_BATCH_SIZE), desc="Embedding batches"):
        batch = text_chunks[i:i + EMBEDDING_BATCH_SIZE]
        texts = [doc.page_content for doc in batch]
        batch_embeddings = embedding_model._client.encode(texts, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)

    texts_and_embeddings = [(doc.page_content, emb) for doc, emb in zip(text_chunks, all_embeddings)]

    db = FAISS.from_embeddings(texts_and_embeddings, embedding_model)
    db.save_local(str(VECTOR_DB_PATH))
    print(f"Vector database created and saved at {VECTOR_DB_PATH}")

    return {
        "documents": len(documents),
        "chunks": len(text_chunks),
        "index_size": db.index.ntotal,
        "dimension": db.index.d,
    }

def update_vector_db(new_documents: List[Document]):
    if not VECTOR_DB_PATH.exists():
        raise FileNotFoundError(f"Vectorstore path does not exist at {VECTOR_DB_PATH}. Please create the vector store first.")

    if not new_documents:
        print("No new documents to incrementally add.")
        return None

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(str(VECTOR_DB_PATH), embedding_model, allow_dangerous_deserialization=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    new_chunks = splitter.split_documents(new_documents)
    print(f"Incrementally adding {len(new_chunks)} new chunks to vectorstore")

    for i in tqdm(range(0, len(new_chunks), EMBEDDING_BATCH_SIZE), desc="Adding chunks"):
        batch = new_chunks[i:i + EMBEDDING_BATCH_SIZE]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        # Pass texts and metadata; embedding computed internally by FAISS wrapper
        db.add_texts(texts, metadatas=metadatas)

    db.save_local(str(VECTOR_DB_PATH))
    print(f"Incremental update complete. Vector database saved at {VECTOR_DB_PATH}")

    return {
        "total_vectors": db.index.ntotal,
        "dimension": db.index.d,
    }

def main():
    # Load processed files log
    processed_files = load_processed_files_log(PROCESSED_LOG_FILE)
    # Determine new or changed files
    new_files = get_new_parquet_files(DATA_DIR, processed_files)

    if not new_files:
        print("No new or modified Parquet files detected; skipping update.")
        return

    print(f"Detected {len(new_files)} new/modified files: {new_files}")

    new_documents = load_parquet_documents(new_files, DATA_DIR)

    if VECTOR_DB_PATH.exists():
        print("Existing vector store found, performing incremental update.")
        stats = update_vector_db(new_documents)
    else:
        print("No vector store found, creating new vector database.")
        stats = create_vector_db(new_documents)

    if stats:
        print("\nVector Database Stats:")
        for k, v in stats.items():
            print(f"{k.replace('_', ' ').title()}: {v}")

        # Update processed_files log with current file modification times
        for file in new_files:
            file_path = os.path.join(DATA_DIR, file)
            processed_files[file] = os.path.getmtime(file_path)
        save_processed_files_log(PROCESSED_LOG_FILE, processed_files)
        print("Processed files log updated.")

if __name__ == "__main__":
    main()
