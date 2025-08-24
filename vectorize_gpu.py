import os
import json
from typing import List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import duckdb
import argparse
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import faiss
from langchain_community.vectorstores import FAISS as LCFAISS

REQUIRED_FIELDS = ["timestamp", "src_ip", "session", "command", "mitre_ttp"]

def parse_args():
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(
        description=(
            "Vectorize Cowrie log files (Parquet) to FAISS (IVFFlat) index + DuckDB.\n"
            "Supports batching by file, row chunk, date, session, and event filters.\n\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data_dir", type=str, default=os.getenv("DATA_DIR", "processed_data"),
                        help="Directory containing .parquet files to process. Default: 'processed_data'")
    parser.add_argument("--vector_db", type=str, default=os.getenv("VECTOR_DB_PATH", "vectorstore/db_faiss.index"),
                        help="Path to save/rebuild the FAISS index (default: 'vectorstore/db_faiss.index')")
    parser.add_argument("--duckdb_path", type=str, default=os.getenv("DUCKDB_PATH", "vectorstore/vector_metadata.duckdb"),
                        help="Path to DuckDB for storing chunk metadata (default: 'vectorstore/vector_metadata.duckdb')")
    parser.add_argument("--model", type=str, default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                        help="Embedding model name or path (default: 'sentence-transformers/all-MiniLM-L6-v2')")
    parser.add_argument("--chunk_size", type=int, default=int(os.getenv("CHUNK_SIZE", "1000")),
                        help="Chunk size (characters) for splitting each document (default: 1000).")
    parser.add_argument("--chunk_overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "50")),
                        help="Overlap size (characters) between chunks (default: 50).")
    parser.add_argument("--embed_batch", type=int, default=int(os.getenv("EMBEDDING_BATCH_SIZE", "256")),
                        help="Batch size for embedding model (default: 256). Lower if RAM limited.")
    parser.add_argument("--process_log", type=str, default=os.getenv("PROCESSED_LOG", "processed_files_log.json"),
                        help="JSON file to keep track of processed files (default: 'processed_files_log.json')")
    parser.add_argument("--nlist", type=int, default=int(os.getenv("NCLUSTERS", "256")),
                        help="Number of FAISS index clusters (nlist). Typical: 256-2048. Default: 256.")
    parser.add_argument("--file_list", type=str, nargs="*",
                        help="Specific Parquet filename(s) (relative to --data_dir) to process.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process this many rows from each file. Use all rows if unset.")
    parser.add_argument("--skip", type=int, default=0,
                        help="Rows to skip from start of each file. Use with --limit for batch.")
    parser.add_argument("--date", type=str,
                        help="Filter by date or range (YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD).")
    parser.add_argument("--session_ids", type=str,
                        help="Comma-separated list of session IDs to embed.")
    parser.add_argument("--event_ids", type=str,
                        help="Comma-separated event IDs to embed.")
    parser.add_argument("--retrain_index", action="store_true",
                        help="Retrain/rebuild the FAISS index using all embeddings in DuckDB.")
    return parser.parse_args()

def load_processed_files_log(log_path: str) -> dict:
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return json.load(f)
    return {}

def save_processed_files_log(log_path: str, data: dict):
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)

def get_new_parquet_files(directory: str, processed_files: dict, file_list: Optional[List[str]] = None) -> List[str]:
    if file_list:
        return [f for f in file_list if os.path.exists(os.path.join(directory, f))]
    all_files = [f for f in os.listdir(directory) if f.endswith(".parquet")]
    new_files = []
    for file in all_files:
        file_path = os.path.join(directory, file)
        mod_time = os.path.getmtime(file_path)
        if file not in processed_files or mod_time > processed_files[file]:
            new_files.append(file)
    return new_files

def parse_date_range(date_str: Optional[str]) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not date_str: return None
    if ':' in date_str:
        start_str, end_str = date_str.split(":", 1)
        return (pd.Timestamp(start_str), pd.Timestamp(end_str))
    else:
        day = pd.Timestamp(date_str)
        return (day, day)

def load_parquet_documents_batched(
    file_paths: List[str],
    directory: str,
    limit: Optional[int] = None,
    skip: int = 0,
    session_ids: Optional[List[str]] = None,
    event_ids: Optional[List[str]] = None,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> List[Document]:
    documents = []
    n_loaded = 0
    for filename in file_paths:
        file_path = os.path.join(directory, filename)
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=1000):
                df = batch.to_pandas()
                if skip:
                    if skip >= len(df):
                        skip -= len(df)
                        continue
                    df = df.iloc[skip:]
                    skip = 0
                if session_ids is not None and 'session' in df.columns:
                    df = df[df['session'].isin(session_ids)]
                if event_ids is not None and 'eventid' in df.columns:
                    df = df[df['eventid'].isin(event_ids)]
                if date_range is not None and 'timestamp' in df.columns:
                    df['ts_date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
                    df = df[(df['ts_date'] >= date_range[0].date()) & (df['ts_date'] <= date_range[1].date())]
                for _, row in df.iterrows():
                    if row is None or row.empty:
                        continue
                    metadata = row.to_dict()
                    # if any(k not in metadata or pd.isna(metadata[k]) for k in REQUIRED_FIELDS):
                    #     continue
                    content_order = ['timestamp', 'session', 'src_ip', 'command', 'mitre_ttp']
                    content_parts = [f"{col}: {metadata[col]}" for col in content_order if col in metadata]
                    for col, value in row.items():
                        if col in content_order:
                            continue  # already added
                        if isinstance(value, (list, np.ndarray)):
                            content_parts.append(f"{col}: {', '.join(map(str, value))}")
                        elif pd.api.types.is_scalar(value) and pd.notna(value):
                            content_parts.append(f"{col}: {value}")
                    page_content = " | ".join(content_parts)
                    for k, v in metadata.items():
                        if isinstance(v, (np.generic, np.ndarray)):
                            metadata[k] = v.item() if np.ndim(v) == 0 else v.tolist()
                    documents.append(Document(page_content=page_content, metadata=metadata))
                    n_loaded += 1
                    if limit is not None and n_loaded >= limit:
                        return documents
        except ImportError:
            print("Error: pyarrow is required for Parquet processing. Please install it.")
            return []
        except Exception as e:
            print(f"[Error] Failed processing {filename}: {str(e)}")
    return documents

def create_vector_db_ivfflat(
    documents: List[Document],
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_batch: int,
    vector_db_dir: Path,
    nlist: int,
):
    if not documents:
        print("No documents to process for vector database creation.")
        return None
    vector_db_dir = Path(vector_db_dir)
    vector_db_dir.mkdir(parents=True, exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    text_chunks = splitter.split_documents(documents)
    print(f"Creating vector database from {len(text_chunks)} chunks (from {len(documents)} documents)")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    all_embeddings = []
    print("Embedding chunks...")
    for i in tqdm(range(0, len(text_chunks), embed_batch), desc="Embedding batches"):
        batch = text_chunks[i:i + embed_batch]
        texts = [doc.page_content for doc in batch]
        batch_embeddings = embedding_model._client.encode(texts, batch_size=embed_batch, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)
        for doc, embedding in zip(batch, batch_embeddings):
            doc.metadata["embedding"] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
    if not all_embeddings:
        print("No embeddings were generated, skipping FAISS index creation.")
        return None

    embeddings = np.array(all_embeddings).astype('float32')
    print(f"Pre-index shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    if embeddings.ndim != 2 or embeddings.shape[1] < 2:
        print(f"ERROR: Embedding shape is {embeddings.shape}. Must be (n_chunks, embedding_dim>=2).")
        return None
    if embeddings.shape == 0:
        print("No embeddings to process for FAISS index.")
        return None
    dim = embeddings.shape[1]
    print(f"FAISS IndexIVFFlat (GPU): training on {embeddings.shape} x {dim}, nlist={nlist}")

    # -- GPU FAISS begins here --
    gpu_res = faiss.StandardGpuResources()
    quantizer = faiss.IndexFlatL2(dim)
    gpu_quantizer = faiss.index_cpu_to_gpu(gpu_res, 0, quantizer)
    # IVF index must be created (and trained) on GPU then saved (via CPU)
    gpu_ivf_index = faiss.IndexIVFFlat(gpu_quantizer, dim, nlist, faiss.METRIC_L2)
    if not gpu_ivf_index.is_trained:
        gpu_ivf_index.train(embeddings)
    gpu_ivf_index.add(embeddings)
    # For saving, move to CPU
    cpu_ivf_index = faiss.index_gpu_to_cpu(gpu_ivf_index)
    faiss.write_index(cpu_ivf_index, str(vector_db_dir / "index.faiss"))

    # Optional: build LangChain store (on CPU for pickle and metadata)
    from langchain_community.vectorstores import FAISS as LCFAISS
    db = LCFAISS.from_documents(
        documents=text_chunks,
        embedding=embedding_model,
    )
    db.save_local(str(vector_db_dir))

    return {
        "documents_processed": len(documents),
        "chunks_created": len(text_chunks),
        "embedding_dimension": dim,
        "faiss_index_type": "IndexIVFFlat",
        "faiss_nlist": nlist,
        "faiss_vectors_added": cpu_ivf_index.ntotal,
    }

def export_metadata_to_duckdb(documents, duckdb_path, table_name="vector_chunks"):
    Path(duckdb_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for doc in documents:
        row = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
        if "embedding" in row and row["embedding"] is not None:
            row['embedding'] = json.dumps(row["embedding"])
        row['page_content'] = getattr(doc, 'page_content', '')
        rows.append(row)
    df = pd.DataFrame(rows)
    def convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_ndarray(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: convert_ndarray(v) for k, v in obj.items()}
        else:
            return obj
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(lambda x: json.dumps(convert_ndarray(x)) if isinstance(x, (dict, list, np.ndarray)) else x)
    con = duckdb.connect(duckdb_path)
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
    con.close()
    print(f"DuckDB metadata exported to {duckdb_path}:{table_name} (rows: {len(df)})")

def load_all_vectors_from_duckdb(duckdb_path, table_name="vector_chunks"):
    con = duckdb.connect(duckdb_path, read_only=True)
    try:
        df = con.execute(f"SELECT embedding FROM {table_name}").fetchdf()
        if df.empty:
            print("No embeddings found in DuckDB!")
            return None
        def to_array(x):
            if isinstance(x, (list, np.ndarray)):
                return np.array(x, dtype=np.float32)
            if isinstance(x, str):
                try:
                    return np.array(json.loads(x), dtype=np.float32)
                except Exception:
                    raise ValueError(f"Embedding JSON decode failed for: {x[:100]}")
            raise TypeError(f"Could not convert {type(x)} to array.")
        emb_list = [to_array(x) for x in df["embedding"]]
        embeddings = np.vstack(emb_list)
        print(f"Loaded {len(embeddings)} vectors from DuckDB.")
        return embeddings
    except Exception as e:
        print(f"Error loading vectors from DuckDB: {e}")
        return None
    finally:
        con.close()

def main(args):
    # Retrain option
    if args.retrain_index:
        print("[REBUILD] Rebuilding and retraining the FAISS index from all vectors in DuckDB...")
        all_embeddings = load_all_vectors_from_duckdb(args.duckdb_path, table_name="vector_chunks")
        min_vectors = args.nlist * 40  # STRONGER minimum
        if all_embeddings is None or len(all_embeddings) < min_vectors:
            print(f"Error: Not enough vectors found for meaningful retraining (found: {0 if all_embeddings is None else len(all_embeddings)}; minimum: {min_vectors}). Consider using more data or check your 'embedding' column.")
            return
        dim = all_embeddings.shape[1]
        print(f"Retraining: Training index with {len(all_embeddings)} vectors and nlist={args.nlist}...")
        gpu_res = faiss.StandardGpuResources()
        quantizer = faiss.IndexFlatL2(dim)
        gpu_quantizer = faiss.index_cpu_to_gpu(gpu_res, 0, quantizer)
        gpu_ivf_index = faiss.IndexIVFFlat(gpu_quantizer, dim, args.nlist, faiss.METRIC_L2)
        gpu_ivf_index.train(all_embeddings)
        gpu_ivf_index.add(all_embeddings)
        cpu_ivf_index = faiss.index_gpu_to_cpu(gpu_ivf_index)
        Path(args.vector_db).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(cpu_ivf_index, str(args.vector_db))
        print(f"Retrained FAISS index written to {args.vector_db}. Done.")
        return
    # Normal/incremental operation
    date_range = parse_date_range(args.date)
    session_ids = args.session_ids.split(",") if args.session_ids else None
    event_ids = args.event_ids.split(",") if args.event_ids else None
    print(f"[CONFIG] Data dir: {args.data_dir}, VectorDB: {args.vector_db}, DuckDB: {args.duckdb_path}, "
          f"Model: {args.model}, Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}, "
          f"Embed batch: {args.embed_batch}, nlist: {args.nlist}, limit: {args.limit}, skip: {args.skip}, "
          f"date: {args.date}, sessions: {session_ids}, events: {event_ids}")
    processed_files = load_processed_files_log(args.process_log)
    new_files = get_new_parquet_files(args.data_dir, processed_files, file_list=args.file_list)
    if not new_files:
        print("No new or selected Parquet files detected; skipping update.")
        return
    print(f"Selected files: {new_files}")
    new_documents = load_parquet_documents_batched(
        new_files, 
        args.data_dir, 
        limit=args.limit,
        skip=args.skip,
        session_ids=session_ids,
        event_ids=event_ids,
        date_range=date_range
    )
    vector_db_dir = Path(args.vector_db)
    if vector_db_dir.suffix:
        vector_db_dir = vector_db_dir.parent / vector_db_dir.stem  # e.g., vectorstore/db_faiss

    print("Creating new LangChain FAISS vectorstore...")
    stats = create_vector_db_ivfflat(
        new_documents, args.model, args.chunk_size, args.chunk_overlap,
        args.embed_batch, vector_db_dir, nlist=args.nlist
    )
    export_metadata_to_duckdb(new_documents, args.duckdb_path, table_name="vector_chunks")
    if stats:
        print("\nVector Database Stats:")
        for k, v in stats.items():
            print(f"{k.replace('_', ' ').title()}: {v}")
        for file in new_files:
            file_path = os.path.join(args.data_dir, file)
            processed_files[file] = os.path.getmtime(file_path)
        save_processed_files_log(args.process_log, processed_files)
        print("Processed files log updated.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
