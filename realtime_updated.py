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
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import duckdb

load_dotenv(find_dotenv())
IPINFO_TOKEN = os.environ.get("IPINFO_TOKEN")

COWRIE_JSON_URL = os.getenv("COWRIE_JSON_URL", "http://115.127.213.187:8311/cowrie.json")
STATE_FILE = os.getenv("COWRIE_PIPELINE_STATE", "pipeline_state.json")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "processed_data")
VECTORDIR = os.getenv("VECTORDB_PATH", "vectorstore/db_faiss")
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "vectorstore/vector_metadata.duckdb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
EMBED_BATCH = int(os.getenv("EMBEDDING_BATCH_SIZE", "512"))
SLEEP_SECONDS = int(os.getenv("COWRIE_PIPELINE_SLEEP", "300"))  # 5min default

# --- Full Regex anomaly detection pattern set from batch processor
ANOMALY_PATTERNS = [
    # Dangerous file/directory deletion and modification
    r"\brm\s+-rf\b", r"\bchmod\s+777\b", r"\bcc\s+/dev/null\b", r"\bchattr\s+\+[a-zA-Z]", r"\bmount\s+\-o\b",
    # Download/dropper/exfil
    r"\bwget\s+https?:\/\/", r"\bcurl\s+https?:\/\/", r"\bftp\s+", r"\btftp\s+", r"\bnc\s+-[el]\b", r"\bsocat\b",
    r"/dev/tcp/", r"/dev/udp/", r"\bnmap\b",
    # Network, pivot, external comms
    r"\bssh\s+", r"\bscp\s+", r"\bsftp\s+", r"\bifconfig\b", r"\blsof\b", r"\bnetstat\b", r"\broute\b",
    r"\bwget\s+-O-", r"\bcurl.*-o",
    # Sudo/priv escalation/exploitation
    r"\b(sudo|su)\b", r"\bsudo\s+-l", r"\bpwgen\b", r"\bsudoedit\b", r"\bpasswd\b",
    # Reverse shell basics
    r"bash\s+-i", r"/bin/bash\s+-i", r"python\s+-c\s*['\"]*import\s+socket.*connect", r"perl\s+-e",
    r"php.*shell_?exec", r"ruby.*socket", r"nc\s+-e", r"powershell.*New-Object.*Net\.Sockets\.TCPClient",
    # Recon/information gathering
    r"\bwhoami\b", r"\bid\b", r"\buname\b", r"\bbanner\b", r"\bhostname\b", r"cat\s+/etc/passwd", r"cat\s+/etc/shadow",
    r"cat\s+/etc/hosts", r"\benv\b", r"\bexport\b", r"\bprintenv\b",
    # Encoding/obfuscation
    r"\bbase64\s+-d\b", r"\bbase64\s+--decode\b", r"eval\s+[`'\"]", r"base32\s+-d", r"base16\s+-d", r"xxd\s+-r",
    r"xxd\s+-p", r"hexdump\s+", r"od\s+-c", r"tr\s+", r"rev", r"gzip\s+-d", r"gunzip\b",
    # Dangerous code exec/automation
    r"\bbusybox\b.*sh", r"\bnohup\b", r"\b&\s*disown\b", r"\bcrontab\b", r"\bat\b\s+now\b", r"\bmkfifo\b",
    # Malware/miner/crypto indicators
    r"\b(minerd?|xmrig|coinhive|crypto|minexmr|gboy|minergate)\b",
    # Common offensive tool references
    r"msfvenom", r"metasploit", r"mimikatz", r"empire", r"csclient", r"pupy", r"cobaltstrike", r"powershell.*-enc",
    # Data exfiltration/deviance
    r"\bwget\s+.*\.exe", r"\bcurl\s+.*\.exe", r"\btcpdump\b", r"\bwireshark\b", r"screencap", r"\bscp\s+.*\.zip\b",
    r"\bscp\s+.*\.tar\b",
    # Misc
    r"\bsendmail\b", r"\bsshd\b.*-o",
]

# --- Full MITRE matrix as per batch
MITRE_MATRIX = {
    'T1087': ['whoami', 'id', 'w', 'getent passwd', 'cat /etc/passwd', 'finger', 'id -u', 'lslogins', 'domain users'],
    'T1033': ['users', 'last', 'groups', 'w', 'who', 'id', 'dsquery', 'net user'],
    'T1082': ['hostname', 'uname', 'lsb_release', 'ver', 'systeminfo', 'hostnamectl'],
    'T1016': ['ifconfig', 'ip addr', 'ip link', 'ipconfig', 'networksetup', 'nmcli', 'netstat', 'route', 'ss -tuln'],
    'T1049': ['arp -a', 'net view', 'nbtstat', 'smbclient'],
    'T1135': ['netstat', 'ss -a', 'ss -tuln', 'lsof', 'tcpdump', 'wireshark'],
    'T1552': ['cat /etc/shadow', 'unshadow', 'find / -name id_rsa', 'ssh-keygen', 'ssh-add', 'cat ~/.ssh/id_rsa', 'extractpassword', 'gpp-decrypt', 'mimikatz', 'procdump'],
    'T1003': ['hashdump', 'cat /etc/shadow', 'secretsdump', 'mimikatz', 'lsass', 'procdump', 'samdump2', 'pwdump'],
    'T1059': ['bash', 'sh', 'python', 'perl', 'php', 'ruby', 'awk', 'powershell', 'pwsh', 'cmd.exe', 'cscript', 'wscript', 'osascript', 'node ', 'jq ', './', 'source ', 'eval ', 'at ', 'cron', 'expect', 'mshta'],
    'T1569': ['systemctl', 'service', '/etc/init.d/', 'killall', 'pkill', 'cron', 'at ', 'schtasks', 'powershell Start-Process'],
    'T1053': ['crontab', 'at ', 'schtasks', 'systemd-timer', 'launchctl'],
    'T1543': ['systemctl enable', 'update-rc.d', 'chkconfig', 'launchctl', 'systemd', 'service add', 'registry add', 'autorun'],
    'T1068': ['sudo', 'su ', 'runas', 'mshta', 'schtasks /run', 'sudoedit', 'pkexec'],
    'T1078': ['ssh ', 'scp ', 'login ', 'rlogin', 'rexec', 'PsExec', 'winrm', 'rdp'],
    'T1070': ['rm -rf', 'shred', 'echo "" > ', 'logrotate --force', 'wipe', 'dd if=', 'cat /dev/null >', 'clear', 'history -c', 'del /f /s /q', 'wevtutil'],
    'T1140': ['base64 ', 'base64 -d', 'openssl enc', 'xxd -r', 'xor ', 'rev ', 'gzip -d', 'gunzip', 'xxd ', 'certutil -decode'],
    'T1056': ['keylogger', 'strace', 'ltrace', 'cat .ssh/known_hosts', 'xinput', 'wireshark', 'sniff', 'tcpdump', 'dumpssp', 'logkeys'],
    'T1021': ['ssh ', 'scp ', 'sftp ', 'telnet ', 'ftp ', 'rlogin ', 'rexec ', 'smbclient', 'pscp', 'psexec', 'winrm', 'rsh '],
    'T1105': ['curl ', 'wget ', 'scp ', 'tftp ', 'ftp ', 'nc -e', 'ncat -e', 'powershell -enc', 'certutil -urlcache', 'bitsadmin', 'Invoke-WebRequest', 'Invoke-Expression', '/dev/tcp/', '/dev/udp/', 'socat ', 'python -c', 'perl -e', 'php -r', 'busybox nc ', 'openbsd-netcat'],
    'T1218': ['regsvr32', 'mshta', 'rundll32', 'wmic', 'bitsadmin', 'cmd.exe /c', 'cmstp', 'InstallUtil', 'msxsl', 'forfiles', 'pubprn'],
    'T1222': ['chmod', 'icacls', 'chown'],
    'T1041': ['curl ', 'scp ', 'ftp ', 'nc ', 'tftp ', 'powershell Out-File', 'tar czf', 'zip ', '7z a'],
    'T1567': ['rsync', 'aws s3 cp', 'azcopy', 'gcloud storage cp'],
    'T1119': ['screencap', 'copy', 'cp ', 'find ', 'cat ', 'dd if=', 'lsattr', 'ls -la', 'lsb_release'],
    'T1113': ['import -window root ', 'xwd ', 'scrot ', 'screencapture'],
    'T1497': ['nmap', 'masscan', 'zmap', 'banner '],
    # ...add any others as in your batch processor, if relevant
}

# --- Enrichment routines matching batch mode ---
def geolocate_ip(ip, geo_cache):
    if not IPINFO_TOKEN or ip.startswith(("10.", "192.168.", "172.")):
        return {}
    if ip in geo_cache:
        return geo_cache[ip]
    try:
        resp = requests.get(f"https://ipinfo.io/{ip}/json?token={IPINFO_TOKEN}", timeout=3)
        if resp.status_code == 200:
            geo = resp.json()
            geo_cache[ip] = {
                'country': geo.get('country', ''),
                'region': geo.get('region', ''),
                'city': geo.get('city', ''),
                'org': geo.get('org', ''),
                'asn': geo.get('asn', ''),
                'location': geo.get('loc', '')
            }
            return geo_cache[ip]
    except Exception:
        pass
    return {}

def detect_anomaly(command):
    if not command:
        return False
    for pattern in ANOMALY_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    return False

def map_mitre(command):
    cmd = (command or '').lower()
    matches = {}
    for tech_id, patterns in MITRE_MATRIX.items():
        if any(p in cmd for p in patterns):
            matches[tech_id] = [p for p in patterns if p in cmd]
    return matches

def enforce_df_schema(df, db_path, table_name="vector_chunks"):
    con = duckdb.connect(db_path)
    schema_cols = [row[1] for row in con.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
    con.close()
    for col in schema_cols:
        if col not in df.columns:
            df[col] = None
    df = df[schema_cols]
    return df

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    else:
        state = {}
    state.setdefault("seen_keys", [])
    state.setdefault("last_byte", 0)
    return state

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def fetch_new_events_stream(url, state):
    print(f"Fetching updates from {url} ...")
    seen_keys = set(state.get("seen_keys", []))
    last_byte = state.get("last_byte", 0)
    if url.startswith("http://") or url.startswith("https://"):
        headers = {'Range': f'bytes={last_byte}-'}
        r = requests.get(url, headers=headers, timeout=60)
        if r.status_code in (200, 206):
            content = r.content
            if last_byte > 0 and len(content) > 0 and content[:1] == b'{':
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
        p = Path(url)
        if not p.exists():
            print(f"[WARN] Local file {url} does not exist.")
            return [], seen_keys, state
        size = p.stat().st_size
        if last_byte > size:
            last_byte = 0
        with open(url, "rb") as f:
            f.seek(last_byte)
            content = f.read()
        state["last_byte"] = last_byte + len(content)
        text = content.decode('utf-8', errors='replace')
    new_lines = [l for l in text.strip().splitlines() if l.strip()]
    print(f"Got {len(new_lines)} new lines")
    return new_lines, seen_keys, state

def enrich_events(lines, session_cache, seen_keys, geo_cache):
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
                continue
            session_id = event.get("session", "")
            src_ip = event.get("src_ip", "")

            # GEO enrichment
            if src_ip:
                event.update(geolocate_ip(src_ip, geo_cache))

            event_type = event.get('eventid', '')

            # Session reconstruction and per-event enrichment
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

            if event_type == 'cowrie.session.connect':
                session_cache[session_id].update({
                    'protocol': event.get('protocol', ''),
                    'dst_ip': event.get('dst_ip', ''),
                    'dst_port': event.get('dst_port', ''),
                    'src_port': event.get('src_port', '')
                })
                event['protocol'] = event.get('protocol', '')
                event['dst_ip'] = event.get('dst_ip', '')
                event['dst_port'] = event.get('dst_port', '')
                event['src_port'] = event.get('src_port', '')

            elif event_type in ('cowrie.login.success', 'cowrie.login.failed'):
                event['username'] = event.get('username', '')
                event['password'] = event.get('password', '')
                event['login_success'] = (event_type == 'cowrie.login.success')
                creds = {
                    'username': event['username'],
                    'password': event['password'],
                    'success': event['login_success'],
                    'timestamp': event['timestamp']
                }
                session_cache[session_id]['credentials'].append(creds)
                if creds['success']:
                    session_cache[session_id]['latest_login'] = creds

            elif event_type == 'cowrie.session.closed':
                if session_id and session_id in session_cache:
                    start_time = session_cache[session_id].get('start_time')
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

            elif event_type == 'cowrie.command.input':
                command = event.get('input', '')
                event['command'] = command
                event['mitre_ttp'] = map_mitre(command)
                event['anomaly_flag'] = detect_anomaly(command)
            else:
                if event_type != 'cowrie.command.input':
                    event['mitre_ttp'] = {}
                    event['anomaly_flag'] = False
                    event['command'] = event.get('input', '')

            if event_type == 'cowrie.session.file_download':
                event['url'] = event.get('url', '')
                event['outfile'] = event.get('outfile', '')
                event['sha256'] = event.get('sha256', '')
                event['size'] = event.get('size', 0)
            elif event_type == 'cowrie.client.kex':
                event['hassh'] = event.get('hassh', '')
                event['kex_algs'] = event.get('kexAlgs', [])
                event['enc_algs'] = event.get('encCS', [])

            # --- Propagate last login/session enrichment into ALL events
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

            # --- Fill all enrichment fields that might be relevant
            for key in ['username', 'password', 'login_success', 'command', 'mitre_ttp', 'anomaly_flag',
                        'connection_duration', 'protocol', 'dst_ip', 'dst_port', 'src_port',
                        'url', 'outfile', 'sha256', 'size', 'hassh', 'kex_algs', 'enc_algs',
                        'country', 'region', 'city', 'org', 'asn', 'location']:
                if key not in event:
                    event[key] = None

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
            if isinstance(v, (list, np.ndarray, dict)):
                content_parts.append(f"{k}: {json.dumps(v)}")
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
    df = enforce_df_schema(df, duckdb_path, table_name)
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
    geo_cache = {}
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
            events, seen_keys = enrich_events(lines, session_cache, seen_keys, geo_cache)
            docs = to_documents(events)
            print(f"Parsed {len(docs)} valid, unique, enriched events.")
            if docs:
                Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
                filename = Path(PROCESSED_DATA_DIR) / f"cowrie_events_inc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                df_save = pd.DataFrame([doc.metadata for doc in docs])
                # Convert lists/dicts/arrays to JSON strings for all object columns
                def convert_obj(val):
                    if isinstance(val, (list, dict, np.ndarray)):
                        return json.dumps(val)
                    return str(val) if val is not None else ""
                for col in df_save.columns:
                    if df_save[col].dtype == "object":
                        df_save[col] = df_save[col].apply(convert_obj)
                # Enforce schema order
                df_save = enforce_df_schema(df_save, DUCKDB_PATH, table_name="vector_chunks")
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
