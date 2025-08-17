import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import sys

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss

def get_path(prompt, default):
    val = input(f"{prompt} [{default}] : ").strip()
    return val or default

def extract_all_texts_and_metadatas(db):
    k = db.index.ntotal
    docs = db.similarity_search("", k=k)
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    return texts, metadatas

def collect_from_many(stores):
    texts, metadatas = [], []
    for store in stores:
        tx, md = extract_all_texts_and_metadatas(store)
        texts.extend(tx)
        metadatas.extend(md)
    return texts, metadatas

def main():
    print("\n--- LANGCHAIN FAISS IVF UPGRADE & (OPTIONAL) MERGE ---\n")
    default_source = "vectorstore/db_faiss"
    default_target = "vectorstore/db_faiss"
    default_model = "sentence-transformers/all-MiniLM-L6-v2"
    default_nlist = "256"

    src_vector_paths = get_path(
        "Enter source vectorstore folder(s) (comma-separated if merging)", default_source)
    out_vectorstore_folder = get_path(
        "Enter output vectorstore folder (will get .faiss and .pkl)", default_target)
    embedding_model_name = get_path("Embedding model name", default_model)
    nlist = int(get_path("IVF nlist (cluster count)", default_nlist))
   
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Parse single/multiple sources
    src_paths = [s.strip() for s in src_vector_paths.split(",") if s.strip()]
    stores = []
    for path in src_paths:
        store = FAISS.load_local(
            path, embedding_model, allow_dangerous_deserialization=True
        )
        print(f"[{path}] contains {store.index.ntotal} vectors.")
        stores.append(store)
    
    # Collect all docs and metadata (from one, or merged from many)
    if len(stores) > 1:
        print(f"Merging {len(stores)} vectorstores ...")
        texts, metadatas = collect_from_many(stores)
    else:
        print(f"Upgrading a single vectorstore ...")
        texts, metadatas = extract_all_texts_and_metadatas(stores[0])
    total_vecs = len(texts)
    print(f"Total chunks to process: {total_vecs}")

    if total_vecs < 2 * nlist:
        print(f"Not enough vectors to upgrade to IVF (need at least {2*nlist}, have {total_vecs}). Exiting.")
        return

    print(f"Re-embedding all texts ...")
    embeddings = []
    EMB_BATCH = 512
    for i in tqdm(range(0, total_vecs, EMB_BATCH), desc="Batch embedding"):
        batch = texts[i:i+EMB_BATCH]
        batch_emb = embedding_model._client.encode(
            batch, batch_size=EMB_BATCH, show_progress_bar=False)
        embeddings.extend(batch_emb)
    embeddings = np.array(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]

    print(f"Training IVF index (nlist={nlist}) ...")
    quantizer = faiss.IndexFlatL2(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)
    ivf.train(embeddings) # type: ignore
    print(f"Adding all vectors to IVF ...")
    ivf.add(embeddings) # type: ignore

    print(f"Building LangChain FAISS vectorstore with new IVF index ...")
    lc_store = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )
    lc_store.index = ivf  # monkeypatch in new IVF

    out_folder = Path(out_vectorstore_folder)
    if out_folder.exists():
        shutil.rmtree(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    print(f"Saving LangChain-compatible FAISS vectorstore to {out_vectorstore_folder}")
    lc_store.save_local(str(out_folder))
    print("\nSUCCESS: LangChain FAISS vectorstore with IVF index saved!")
    print(f"Files generated: {list(out_folder.iterdir())}")
    print(f"IVF index summary: nlist={ivf.nlist}, ntotal={ivf.ntotal}, is_trained={ivf.is_trained}")

if __name__ == "__main__":
    main()
