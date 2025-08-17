import os
import json
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import duckdb

load_dotenv(find_dotenv())

# --- Configuration
COWRIE_JSON_URL = os.getenv("COWRIE_JSON_URL", "http://115.127.213.187:8080/cowrie.json")
STATE_FILE = os.getenv("COWRIE_PIPELINE_STATE", "pipeline_state.json")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "processed_data")
VECTORDIR = os.getenv("VECTORDB_PATH", "vectorstore/db_faiss")
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "vectorstore/vector_metadata.duckdb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
EMBED_BATCH = int(os.getenv("EMBEDDING_BATCH_SIZE", "512"))
SLEEP_SECONDS = int(os.getenv("COWRIE_PIPELINE_SLEEP", "300"))  # 5min default

# --- Anomaly and MITRE Mapping
ANOMALY_COMMANDS = [
    'rm -rf /', 'wget http', 'curl http', 'nc -l', 'netcat', 'exec ', 'base64 -d', 'chmod 777',
    'ssh-keyscan', 'ssh ', 'scp ', 'telnet ', 'ftp ', 'sftp ', 'python -c', 'perl -e', 'eval '
]
MITRE_MATRIX = {
    'T1087': ['whoami', 'id', 'w', 'getent passwd', 'cat /etc/passwd', 'finger', 'id -u'],
    'T1059': ['bash', 'sh', 'python', 'perl', 'php', 'awk', r'./', 'powershell', 'cmd.exe', 'cscript'],
    'T1552': ['unshadow', 'cat /etc/shadow', 'find / -name id_rsa', 'ssh-keygen', 'ssh-add'],
    'T1021': ['ssh ', 'scp ', 'telnet ', 'ftp ', 'sftp ', 'rlogin', 'rexec'],
    'T1070': ['rm -rf', 'shred', 'echo "" > ', 'logrotate --force', 'wipe', 'dd if=', 'cat /dev/null >'],
    'T1056': ['keylogger', 'strace', 'ltrace', 'cat .ssh/known_hosts', 'xinput', 'wireshark'],
    'T1569': ['systemctl', 'service', '/etc/init.d/', 'killall', 'pkill', 'cron', 'at '],
    'T1003': ['hashdump', 'mimikatz', 'lsass', 'procdump'],
    'T1135': ['net view', 'netsh', 'netstat', 'arp -a'],
    'T1218': ['regsvr32', 'mshta', 'rundll32'],
    'T1105': ['curl', 'wget', 'scp'],
}

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    else:
        state = {}
    # Ensure new keys for restarts
    state.setdefault("seen_keys", [])
    state.setdefault("last_byte", 0)
    return state

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def fetch_new_events_stream(url, state):
    """
    Efficiently fetch only new data from the log since the last processed byte.
    Supports both HTTP and local file paths transparently.
    Returns: list of new lines, previous seen_keys set, updated state
    """
    print(f"Fetching updates from {url} ...")
    seen_keys = set(state.get("seen_keys", []))
    last_byte = state.get("last_byte", 0)

    if url.startswith("http://") or url.startswith("https://"):
        headers = {'Range': f'bytes={last_byte}-'}
        r = requests.get(url, headers=headers, timeout=60)
        if r.status_code in (200, 206):  # 206 = partial content
            content = r.content
            # If log rotated, or last_byte invalid (file truncated), server may send from start
            if last_byte > 0 and len(content) > 0 and content[:1] == b'{':
                # Heuristic: most logs won't start with a { mid-line
                print("Detected possible log rotation—resetting offset!")
                last_byte = 0
                headers = {'Range': 'bytes=0-'}
                r = requests.get(url, headers=headers, timeout=60)
                content = r.content
            state["last_byte"] = last_byte + len(content)
        else:
            print(f"HTTP failed with {r.status_code}—assuming log is gone or moved. Sleeping.")
            return [], seen_keys, state
        text = content.decode('utf-8', errors='replace')
    else:
        # LOCAL FILE
        p = Path(url)
        if not p.exists():
            print(f"[WARN] Local file {url} does not exist.")
            return [], seen_keys, state
        size = p.stat().st_size
        if last_byte > size:  # log rotated/truncated
            last_byte = 0
        with open(url, "rb") as f:
            f.seek(last_byte)
            content = f.read()
            state["last_byte"] = last_byte + len(content)
        text = content.decode('utf-8', errors='replace')

    # Only yield non-empty, new lines
    new_lines = [l for l in text.strip().splitlines() if l.strip()]
    print(f"Got {len(new_lines)} new lines")
    return new_lines, seen_keys, state

def is_anomaly(command):
    if not command:
        return False
    cmd_l = command.lower()
    return any(anom in cmd_l for anom in ANOMALY_COMMANDS)

def ttp_mapping(command):
    if not command:
        return {}
    cmd_l = command.lower()
    matches = {}
    for tech_id, patterns in MITRE_MATRIX.items():
        if any(p in cmd_l for p in patterns):
            matches[tech_id] = [p for p in patterns if p in cmd_l]
    return matches

def enrich_events(lines, session_cache, seen_keys):
    events = []
    for line in lines:
        try:
            event = json.loads(line.strip())
            unique_key = (
                str(event.get("session", "")) + "|" +
                str(event.get("eventid", "")) + "|" +
                str(event.get("timestamp", ""))
            )
            if unique_key in seen_keys:
                continue  # Safe deduplication

            # Session reconstruction (stateless between pipeline restarts)
            session_id = event.get("session", "")
            if session_id and session_id not in session_cache:
                session_cache[session_id] = {
                    'start_time': event.get('timestamp'),
                    'end_time': None,
                    'protocol': event.get('protocol', ''),
                    'dst_ip': event.get('dst_ip', ''),
                    'dst_port': event.get('dst_port', ''),
                    'src_port': event.get('src_port', ''),
                    'credentials': [],
                    'latest_login': {},
                    'connection_duration': None
                }
            event_type = event.get('eventid', '')
            if event_type == 'cowrie.session.connect':
                session_cache[session_id].update({
                    'protocol': event.get('protocol', ''),
                    'dst_ip': event.get('dst_ip', ''),
                    'dst_port': event.get('dst_port', ''),
                    'src_port': event.get('src_port', '')
                })
            elif event_type in ('cowrie.login.success', 'cowrie.login.failed'):
                creds = {
                    'username': event.get('username', ''),
                    'password': event.get('password', ''),
                    'success': (event_type == 'cowrie.login.success'),
                    'timestamp': event['timestamp']
                }
                session_cache[session_id]['credentials'].append(creds)
                if creds['success']:
                    session_cache[session_id]['latest_login'] = creds
            elif event_type == 'cowrie.session.closed':
                if session_id in session_cache:
                    s_cache = session_cache[session_id]
                    start_time = s_cache.get('start_time')
                    end_time = event['timestamp']
                    if start_time:
                        try:
                            duration = (datetime.fromisoformat(end_time.rstrip('Z')) -
                                        datetime.fromisoformat(start_time.rstrip('Z'))).total_seconds()
                            session_cache[session_id]['end_time'] = end_time
                            session_cache[session_id]['connection_duration'] = duration
                            event['connection_duration'] = duration
                        except Exception:
                            pass

            # Add enrichment—MITRE, anomaly
            if event_type == "cowrie.command.input":
                event["command"] = event.get("input", "")
            event["mitre_ttp"] = ttp_mapping(event.get("command", ""))
            event["anomaly_flag"] = is_anomaly(event.get("command", ""))
            # Propagate session info
            if session_id in session_cache:
                s_cache = session_cache[session_id]
                llogin = s_cache.get("latest_login", {})
                if llogin:
                    event["username"] = llogin.get("username", "")
                    event["password"] = llogin.get("password", "")
                    event["login_success"] = llogin.get("success", False)
                event["connection_duration"] = s_cache.get("connection_duration")
                event["protocol"] = s_cache.get("protocol", "")
                event["dst_ip"] = s_cache.get("dst_ip", "")
                event["dst_port"] = s_cache.get("dst_port", "")
                event["src_port"] = s_cache.get("src_port", "")
            event["_unique_key"] = unique_key
            events.append(event)
            seen_keys.add(unique_key)
        except Exception as e:
            print(f"Skipping invalid event: {e} ({line[:80]}...)")
    return events, seen_keys

def to_documents(events):
    docs = []
    for event in events:
        content_parts = []
        for k, v in event.items():
            if isinstance(v, (list, np.ndarray)):
                content_parts.append(f"{k}: {', '.join(map(str, v))}")
            elif pd.api.types.is_scalar(v) and pd.notna(v):
                content_parts.append(f"{k}: {v}")
        page_content = " | ".join(content_parts)
        docs.append(Document(page_content=page_content, metadata=event))
    return docs

def vectorstore_update(new_documents, vector_db_path, embedding_model_name, chunk_size, chunk_overlap, embed_batch):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_db_path = Path(vector_db_path)
    index_faiss_path = vector_db_path / "index.faiss"
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    new_chunks = splitter.split_documents(new_documents)
    
    if index_faiss_path.exists():
        db = FAISS.load_local(str(vector_db_path), embedding_model, allow_dangerous_deserialization=True)
        print(f"Loaded FAISS vectorstore, adding {len(new_chunks)} chunks...")
        for i in tqdm(range(0, len(new_chunks), embed_batch), desc="Adding vectors"):
            batch = new_chunks[i:i + embed_batch]
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            db.add_texts(texts, metadatas=metadatas)
        db.save_local(str(vector_db_path))
        print(f"FAISS vectorstore updated.")
    else:
        print("No FAISS vectorstore found, creating fresh one.")
        all_embeddings = []
        for i in tqdm(range(0, len(new_chunks), embed_batch), desc="Embedding vectors"):
            batch = new_chunks[i:i + embed_batch]
            texts = [doc.page_content for doc in batch]
            batch_embeddings = embedding_model._client.encode(texts, batch_size=embed_batch, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings)
        texts_and_embeddings = [(doc.page_content, emb) for doc, emb in zip(new_chunks, all_embeddings)]
        db = FAISS.from_embeddings(texts_and_embeddings, embedding_model)
        db.save_local(str(vector_db_path))
        print(f"FAISS vectorstore created.")
    return len(new_chunks)

def append_to_duckdb(docs, duckdb_path, table_name="vector_chunks"):
    rows = []
    for doc in docs:
        row = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
        row['page_content'] = getattr(doc, 'page_content', '')
        rows.append(row)
    if not rows:
        print("No rows for DuckDB.")
        return 0
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
    con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df LIMIT 0")
    con.execute(f"INSERT INTO {table_name} SELECT * FROM df")
    added = len(df)
    con.close()
    print(f"DuckDB: appended {added} rows.")
    return added

def main():
    print("[Realtime pipeline initializing: streaming mode!]")
    state = load_state()
    seen_keys = set(state.get("seen_keys", []))
    session_cache = {}

    while True:
        try:
            lines, prev_known, state = fetch_new_events_stream(COWRIE_JSON_URL, state)
            seen_keys.update(prev_known)
            if not lines:
                print("No new events, sleeping...")
                state["seen_keys"] = list(seen_keys)
                save_state(state)
                time.sleep(SLEEP_SECONDS)
                continue
            events, seen_keys = enrich_events(lines, session_cache, seen_keys)
            docs = to_documents(events)
            print(f"Parsed {len(docs)} valid, unique, enriched events.")
            if docs:
                Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
                filename = Path(PROCESSED_DATA_DIR) / f"cowrie_events_inc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                df_save = pd.DataFrame([doc.metadata for doc in docs])

                # --- FIX: Convert lists/dicts/arrays to JSON strings for all object columns, especially 'message' ---
                def convert_obj(val):
                    if isinstance(val, (list, dict, np.ndarray)):
                        return json.dumps(val)
                    return str(val) if val is not None else ""

                for col in df_save.columns:
                    if df_save[col].dtype == "object":
                        df_save[col] = df_save[col].apply(convert_obj)

                df_save.to_parquet(filename)
                print(f"New events written to: {filename}")
            vectorstore_update(docs, VECTORDIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, EMBED_BATCH)
            append_to_duckdb(docs, DUCKDB_PATH)
            print(f"Run complete. Sleeping {SLEEP_SECONDS}s ...")
            state["seen_keys"] = list(seen_keys)
            save_state(state)
            time.sleep(SLEEP_SECONDS)
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break
        except Exception as e:
            print(f"Top-level error: {e}")
            time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
