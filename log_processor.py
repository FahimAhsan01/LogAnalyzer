import os
import json
import logging
import re
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Union, Generator, Tuple
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


load_dotenv(find_dotenv())
IPINFO_TOKEN = os.environ.get("IPINFO_TOKEN")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LOCAL_STATE_LOG = "processed_logs_state.json"
REMOTE_STATE_LOG = "remote_logs_state.json"

def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")

def line_hash(line: str) -> str:
    return hashlib.sha1(line.encode("utf-8")).hexdigest()

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parse one file or all files in a directory with session, TTP, anomaly, and geo enrichment.\n"
            "\n"
            "Batch and incremental mode: only process new lines/events from each log (progress saved), filter by line, date, session, or event.\n"
            "\n"
            "EXAMPLES:\n"
            "  # Parse local or remote Cowrie logs (incrementally, deduplicated) with geo/session/anomaly enrichment.\n\n"
            "  EXAMPLES:\n"
            "  python log_processor.py cowrie.json.2025-07-31\n"
            "  python log_processor.py ./logs --date YYYY-MM-DD\n"
            "  python log_processor.py http://192.168.0.1:8080/cowrie.json\n"
            "  # Parse all files in a directory, only events from 2025-08-15\n"
            "  python log_processor.py ./logs --date 2025-08-15\n"
            "  # Parse only sessions S1,S2 in lines 20000-39999 (incremental)\n"
            "  python log_processor.py mycowrie.json --skip 20000 --limit 20000 --session_ids S1,S2\n"
            "  # Disable geolocation enrichment and use CSV output\n"
            "  python log_processor.py mycowrie.json --no_geo --output_format csv\n"
            "  # Force reprocessing everything, erase progress log\n"
            "  python log_processor.py ./logs --force\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_path", type=str,
        help="Local path or HTTP(S) URL to a JSON log, or a directory of such files.")
    parser.add_argument("--skip", type=int, default=0,
        help="Number of lines to skip at start of each log file (for chunking/resume).")
    parser.add_argument("--limit", type=int, default=None,
        help="Maximum number of log lines to process (after skip) in each log file. Use for batch processing of large logs.")
    parser.add_argument("--date", type=str,
        help="Filter: only include events from this date or range (YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD).")
    parser.add_argument("--session_ids", type=str,
        help="Comma-separated list of session IDs to include. Example: SESS123,SESS456")
    parser.add_argument("--event_ids", type=str,
        help="Comma-separated list of event IDs to include. Example: cowrie.command.input,cowrie.session.closed")
    parser.add_argument("--output_dir", type=str, default=os.getenv("COWRIE_OUTPUT_DIR", "processed_data"),
        help="Directory to write processed output files. Default: processed_data")
    parser.add_argument("--output_format", type=str, default=os.getenv("COWRIE_OUTPUT_FORMAT", "parquet"),
        choices=["parquet", "csv", "jsonl", "json", "jsonlines"],
        help="Output format for parsed events [parquet|csv|jsonl|json|jsonlines]. Recommended: parquet.")
    parser.add_argument("--max_workers", type=int, default=int(os.getenv("COWRIE_MAX_WORKERS", "4")),
        help="Number of threads for parallel batch parsing (directory mode). Default: 4")
    parser.add_argument("--no_geo", action="store_true",
        help="Disable IP geolocation enrichment (faster; attacker location not added).")
    parser.add_argument("--force", action="store_true",
        help="Force reprocessing of all input files, ignoring previous progress log.")
    return parser.parse_args()

def parse_date_range(date_str: Optional[str]) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not date_str: return None
    if ':' in date_str:
        start_str, end_str = date_str.split(":", 1)
        return (pd.Timestamp(start_str), pd.Timestamp(end_str))
    else:
        day = pd.Timestamp(date_str)
        return (day, day)

def load_state(state_log_path: Path) -> dict:
    if state_log_path.is_file():
        with open(state_log_path, "r") as f:
            return json.load(f)
    return {}

def save_state(state_log_path: Path, state: dict):
    with open(state_log_path, "w") as f:
        json.dump(state, f, indent=2)

def filter_dataframe(df: pd.DataFrame,
                    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]]=None,
                    session_ids: Optional[List[str]]=None,
                    event_ids: Optional[List[str]]=None,
                    ) -> pd.DataFrame:
    if date_range and "timestamp" in df.columns:
        df['ts_date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
        df = df[(df['ts_date'] >= date_range[0].date()) & (df['ts_date'] <= date_range[1].date())]

    if session_ids and 'session' in df.columns:
        df = df[df['session'].isin(session_ids)]
    if event_ids and 'eventid' in df.columns:
        df = df[df['eventid'].isin(event_ids)]
    return df

def stream_remote_log_incremental(url: str, last_line_hash: Optional[str] = None) -> Generator[str, None, None]:
    seen = last_line_hash is not None
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                h = line_hash(line)
                if not seen:
                    yield line
                elif h == last_line_hash:
                    seen = False  # Found the last already processed line, next is new
    except Exception as e:
        logger.error(f"Error streaming remote log {url}: {e}")
class CowrieJSONProcessor:
    """Optimized parser for Cowrie JSON logs with session reconstruction and enhanced features"""

    def __init__(self, output_dir: str = "processed_data", output_format: str = "parquet", max_workers: int = 4, geo_enabled: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.geo_cache = {}
        self.session_cache = {}  # Track sessions across events
        self.attack_matrix = self._load_mitre_matrix()
        self.output_format = output_format.lower()
        self.max_workers = max_workers
        self.geo_enabled = True
# Comprehensive patterns for attacker and post-exploitation behavior matching
        self.anomaly_patterns = [
            # Dangerous file/directory deletion and modification
            r"\brm\s+-rf\b",          # rm -rf destructive delete
            r"\bchmod\s+777\b",       # World-writable, commonly for dropped binaries/scripts
            r"\bcc\s+/dev/null\b",    # Hide output
            r"\bchattr\s+\+[a-zA-Z]", # Change file attributes (rootkits)
            r"\bmount\s+\-o\b",       # Malicious mounting

            # Download, dropper, exfil
            r"\bwget\s+https?:\/\/",  # wget URL
            r"\bcurl\s+https?:\/\/",  # curl URL
            r"\bftp\s+", r"\btftp\s+", # Legacy downloads
            r"\bnc\s+-[el]\b",        # Netcat with execute or listen
            r"\bsocat\b",             # Reverse shells, relay
            r"/dev/tcp/", r"/dev/udp/", # Shell via /dev/tcp/host/port
            r"\bnmap\b",              # Reconnaissance/port scanning

            # Network, pivot, and external comms
            r"\bssh\s+", r"\bscp\s+", r"\bsftp\s+",     # Data exfil or lateral movement
            r"\bifconfig\b", r"\blsof\b", r"\bnetstat\b", r"\broute\b", # Discovery/pivot
            r"\bwget\s+-O-", r"\bcurl.*-o",             # Write output to file/pipe

            # Sudo/privilege escalation/exploitation
            r"\b(sudo|su)\b",          # Attempt elevation
            r"\bsudo\s+-l",            # List allowed commands
            r"\bpwgen\b",              # Password guessing
            r"\bsudoedit\b",           # Sudoedit abuse
            r"\bpasswd\b",             # Change passwords

            # Reverse shell basics (bash, python, perl, etc.)
            r"bash\s+-i", r"/bin/bash\s+-i",
            r"python\s+-c\s*['\"]*import\s+socket.*connect",
            r"perl\s+-e",
            r"php.*shell_?exec", r"ruby.*socket", r"nc\s+-e",
            r"powershell.*New-Object.*Net\.Sockets\.TCPClient", # powershell RCE

            # Recon/information gathering
            r"\bwhoami\b", r"\bid\b", r"\buname\b", r"\bbanner\b", r"\bhostname\b",
            r"cat\s+/etc/passwd", r"cat\s+/etc/shadow", r"cat\s+/etc/hosts",
            r"\benv\b", r"\bexport\b", r"\bprintenv\b",

            # Encoding/obfuscation
            r"\bbase64\s+-d\b", r"\bbase64\s+--decode\b",
            r"eval\s+[`'\"]", r"base32\s+-d", r"base16\s+-d",
            r"xxd\s+-r", r"xxd\s+-p", r"hexdump\s+", r"od\s+-c", r"tr\s+", r"rev",
            r"gzip\s+-d", r"gunzip\b",

            # Dangerous code exec and automation
            r"\bbusybox\b.*sh",  # Busybox shell escape
            r"\bnohup\b",        # Detach job
            r"\b&\s*disown\b",   # Detach shell
            r"\bcrontab\b",      # Persistence
            r"\bat\b\s+now\b",   # Scheduled task
            r"\bmkfifo\b",       # Named pipe tricks
            r"\bnohup\b",

            # Malware/crypto-miner indicators
            r"\b(minerd?|xmrig|coinhive|crypto|minexmr|gboy|minergate)\b",

            # Common offensive tool references
            r"msfvenom", r"metasploit", r"mimikatz", r"empire", r"csclient",
            r"pupy", r"cobaltstrike", r"powershell.*-enc",

            # Data exfiltration and deviance
            r"\bwget\s+.*\.exe", r"\bcurl\s+.*\.exe",       # Suspicious downloads
            r"\btcpdump\b", r"\bwireshark\b", r"screencap", # Packet capture, screenshot
            r"\bscp\s+.*\.zip\b", r"\bscp\s+.*\.tar\b"      # Bulk data transfer

            # Misc
            r"\bsendmail\b", r"\bsshd\b.*-o", # Backdoored sshd
        ]

    def _load_mitre_matrix(self) -> Dict[str, List[str]]:
        """Expanded MITRE ATT&CK patterns with technique IDs and wide coverage."""
        # Sources: MITRE ATT&CK, common CTF/pentest/honeypot behaviors, Sigma, and real-world TTPs.
        # Patterns are lowercase for easy string matching.
        return {
            # Discovery
            'T1087': ['whoami', 'id', 'w', 'getent passwd', 'cat /etc/passwd', 'finger', 'id -u', 'lslogins', 'domain users'],
            'T1033': ['users', 'last', 'groups', 'w', 'who', 'id', 'dsquery', 'net user'],
            'T1082': ['hostname', 'uname', 'lsb_release', 'ver', 'systeminfo', 'hostnamectl'],  # System info
            'T1016': ['ifconfig', 'ip addr', 'ip link', 'ipconfig', 'networksetup', 'nmcli', 'netstat', 'route', 'ss -tuln'],  # Network
            'T1049': ['arp -a', 'net view', 'nbtstat', 'smbclient'],  # Network discovery/smb
            'T1135': ['netstat', 'ss -a', 'ss -tuln', 'lsof', 'tcpdump', 'wireshark'],

            # Credential Access
            'T1552': ['cat /etc/shadow', 'unshadow', 'find / -name id_rsa', 'ssh-keygen', 'ssh-add', 'cat ~/.ssh/id_rsa', 'extractpassword', 'gpp-decrypt', 'mimikatz', 'procdump'],
            'T1003': ['hashdump', 'cat /etc/shadow', 'secretsdump', 'mimikatz', 'lsass', 'procdump', 'samdump2', 'pwdump'],

            # Execution and Scripting
            'T1059': [
                'bash', 'sh', 'python', 'perl', 'php', 'ruby', 'awk', 'powershell', 'pwsh', 'cmd.exe', 'cscript', 'wscript', 'osascript', 'node ', 'jq ', './', 'source ', 'eval ', 'at ', 'cron', 'expect', 'mshta'
            ],
            'T1569': ['systemctl', 'service', '/etc/init.d/', 'killall', 'pkill', 'cron', 'at ', 'schtasks', 'powershell Start-Process'],  # Service execution
            'T1053': ['crontab', 'at ', 'schtasks', 'systemd-timer', 'launchctl'],  # Scheduled task

            # Persistence
            'T1543': ['systemctl enable', 'update-rc.d', 'chkconfig', 'launchctl', 'systemd', 'service add', 'registry add', 'autorun'],  # Auto-start

            # Privilege Escalation
            'T1068': ['sudo', 'su ', 'runas', 'mshta', 'schtasks /run', 'sudoedit', 'pkexec'],  # Priv esc
            'T1078': ['ssh ', 'scp ', 'login ', 'rlogin', 'rexec', 'PsExec', 'winrm', 'rdp'],  # Valid accounts/lateral move

            # Defense Evasion, Clearing Logs
            'T1070': ['rm -rf', 'shred', 'echo "" > ', 'logrotate --force', 'wipe', 'dd if=', 'cat /dev/null >', 'clear', 'history -c', 'del /f /s /q', 'wevtutil'],
            'T1140': ['base64 ', 'base64 -d', 'openssl enc', 'xxd -r', 'xor ', 'rev ', 'gzip -d', 'gunzip'],  # Decoding/Obfuscation

            # Input Capture / Keyloggers
            'T1056': [
                'keylogger', 'strace', 'ltrace', 'cat .ssh/known_hosts', 'xinput', 'wireshark', 'sniff', 'tcpdump', 'dumpssp', 'logkeys'
            ],

            # Lateral Movement
            'T1021': ['ssh ', 'scp ', 'sftp ', 'telnet ', 'ftp ', 'rlogin ', 'rexec ', 'smbclient', 'pscp', 'psexec', 'winrm', 'rsh '],

            # Command and Control / Ingress tool transfer
            'T1105': [
                'curl ', 'wget ', 'scp ', 'tftp ', 'ftp ', 'nc -e', 'ncat -e', 'powershell -enc', 'certutil -urlcache', 'bitsadmin', 'Invoke-WebRequest', 'Invoke-Expression',
                '/dev/tcp/', '/dev/udp/', 'socat ', 'python -c', 'perl -e', 'php -r', 'busybox nc ', 'openbsd-netcat'
            ],

            # Masquerading / Living off the land binaries (LOLBins)
            'T1218': [
                'regsvr32', 'mshta', 'rundll32', 'wmic', 'bitsadmin', 'cmd.exe /c', 'cmstp', 'InstallUtil', 'msxsl', 'forfiles', 'pubprn'
            ],
            'T1222': ['chmod', 'icacls', 'chown'],  # File permission

            # Exfiltration
            'T1041': ['curl ', 'scp ', 'ftp ', 'nc ', 'tftp ', 'powershell Out-File', 'tar czf', 'zip ', '7z a'],
            'T1567': ['rsync', 'aws s3 cp', 'azcopy', 'gcloud storage cp'],

            # Collection 
            'T1119': ['screencap', 'copy', 'cp ', 'find ', 'cat ', 'dd if=', 'lsattr', 'ls -la', 'lsb_release'],
            'T1113': ['import -window root ', 'xwd ', 'scrot ', 'screencapture'],

            # Other
            'T1497': ['nmap', 'masscan', 'zmap', 'banner '],  # Network scanning/fingerprinting
            'T1140': ['base64 ', 'xxd ', 'openssl enc', 'certutil -decode']  # Encoding/obfuscation
        }


    def _geolocate_ip(self, ip: str) -> Dict[str, str]:
        """Cached IP geolocation with error handling"""
        if not self.geo_enabled:
            return {}
        
        if ip in self.geo_cache:
            return self.geo_cache[ip]

        if not IPINFO_TOKEN or ip.startswith(("10.", "192.168.", "172.")):
            return {}

        try:
            resp = requests.get(
                f"https://ipinfo.io/{ip}/json?token={IPINFO_TOKEN}",
                timeout=3
            )
            if resp.status_code == 200:
                geo = resp.json()
                self.geo_cache[ip] = {
                    'country': geo.get('country', ''),
                    'region': geo.get('region', ''),
                    'city': geo.get('city', ''),
                    'org': geo.get('org', ''),
                    'asn': geo.get('asn', ''),
                    'location': geo.get('loc', '')
                }
                return self.geo_cache[ip]
        except Exception as e:
            logger.debug(f"Geolocation failed for {ip}: {str(e)}")
        return {}

    def _categorize_ttp(self, command: str) -> Dict[str, List[str]]:
        """Map commands to MITRE techniques"""
        command_lower = command.lower()
        matches = {}
        for tech_id, patterns in self.attack_matrix.items():
            if any(p in command_lower for p in patterns):
                matches[tech_id] = [p for p in patterns if p in command_lower]
        return matches

    def _detect_anomaly(self, command: str) -> bool:
        command = command.lower()
        for pattern in self.anomaly_patterns:
            if re.search(pattern, command):
                return True
        return False
    
    def _process_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Core event processing with session awareness, enriched metadata, and anomaly detection"""
        processed = {
            'timestamp': datetime.fromisoformat(event['timestamp'].rstrip('Z')),
            'event_type': event['eventid'],
            'session': event.get('session', ''),
            'src_ip': event.get('src_ip', ''),
            'sensor': event.get('sensor', ''),
            'username': None,
            'password': None,
            'login_success': False,
            'command': None,
            'mitre_ttp': {},
            'anomaly_flag': False,
            'connection_duration': None,
            'protocol': None,
            'dst_ip': None,
            'dst_port': None,
            'src_port': None,
            # file download specific:
            'url': None,
            'outfile': None,
            'sha256': None,
            'size': None,
            # encryption:
            'hassh': None,
            'kex_algs': None,
            'enc_algs': None
        }

        # Add geolocation info if applicable
        if processed['src_ip']:
            processed.update(self._geolocate_ip(processed['src_ip']))

        session_id = processed['session']

        # Initialize session cache if new
        if session_id and session_id not in self.session_cache:
            self.session_cache[session_id] = {
                'start_time': processed['timestamp'],
                'end_time': None,
                'protocol': '',
                'dst_ip': '',
                'dst_port': '',
                'src_port': '',
                'credentials': [],
                'latest_login': {},  # latest successful login creds
                'connection_duration': None
            }

        event_type = processed['event_type']

        if event_type == 'cowrie.session.connect':
                self.session_cache[session_id].update({
                    'protocol': event.get('protocol', ''),
                    'dst_ip': event.get('dst_ip', ''),
                    'dst_port': event.get('dst_port', ''),
                    'src_port': event.get('src_port', '')
                })
                processed['protocol'] = event.get('protocol', '')
                processed['dst_ip'] = event.get('dst_ip', '')
                processed['dst_port'] = event.get('dst_port', '')
                processed['src_port'] = event.get('src_port', '')

        elif event_type in ('cowrie.login.success', 'cowrie.login.failed'):
            # Save ALL credentials regardless of success/failure
            processed['username'] = event.get('username', '')
            processed['password'] = event.get('password', '')
            processed['login_success'] = (event_type == 'cowrie.login.success')

            creds = {
                'username': processed['username'],
                'password': processed['password'],
                'success': processed['login_success'],
                'timestamp': processed['timestamp']
            }
            self.session_cache[session_id]['credentials'].append(creds)
            if creds['success']:
                self.session_cache[session_id]['latest_login'] = creds


        elif event_type == 'cowrie.session.closed':
            if session_id and session_id in self.session_cache:
                start_time = self.session_cache[session_id].get('start_time')
                end_time = processed['timestamp']
                if start_time:
                    duration = (end_time - start_time).total_seconds()
                    self.session_cache[session_id]['end_time'] = end_time
                    self.session_cache[session_id]['connection_duration'] = duration
                    processed['connection_duration'] = duration

        elif event_type == 'cowrie.command.input':
            command = event.get('input', '')
            processed['command'] = command
            processed['mitre_ttp'] = self._categorize_ttp(command)
            processed['anomaly_flag'] = self._detect_anomaly(command)

        elif event_type == 'cowrie.session.file_download':
            processed['url'] = event.get('url', '')
            processed['outfile'] = event.get('outfile', '')
            processed['sha256'] = event.get('sha256', '')
            processed['size'] = event.get('size', 0)

        elif event_type == 'cowrie.client.kex':
            processed['hassh'] = event.get('hassh', '')
            processed['kex_algs'] = event.get('kexAlgs', [])
            processed['enc_algs'] = event.get('encCS', [])

        # --- Propagate latest login info and session metadata into every event ---
        if session_id and session_id in self.session_cache:
            latest_login = self.session_cache[session_id].get('latest_login', {})
            if latest_login:
                processed['username'] = latest_login.get('username', '')
                processed['password'] = latest_login.get('password', '')
                processed['login_success'] = latest_login.get('success', False)

            processed['connection_duration'] = self.session_cache[session_id].get('connection_duration')
            processed['protocol'] = self.session_cache[session_id].get('protocol', '')
            processed['dst_ip'] = self.session_cache[session_id].get('dst_ip', '')
            processed['dst_port'] = self.session_cache[session_id].get('dst_port', '')
            processed['src_port'] = self.session_cache[session_id].get('src_port', '')

        return processed

    def parse_file(self, file_path: Path) -> pd.DataFrame:
        """Parse single JSON log file with progress tracking"""
        if not file_path.is_file():
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()

        events = []
        total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))

        with open(file_path, 'r', encoding='utf-8') as f, \
             tqdm(total=total_lines, desc=f"Parsing {file_path.name}") as pbar:

            for line in f:
                try:
                    json_event = json.loads(line.strip())
                    if parsed := self._process_event(json_event):
                        events.append(parsed)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {line[:100]}...")
                pbar.update(1)

        df = pd.DataFrame(events)
        return df

    def parse_files_parallel(self, dir_path: Path) -> pd.DataFrame:
        """Parse multiple files in parallel using ThreadPoolExecutor for faster processing"""
        if not dir_path.is_dir():
            logger.error(f"Not a directory: {dir_path}")
            return pd.DataFrame()

        date_pattern = re.compile(r"\d{4}[-_]\d{2}[-_]\d{2}$")
        files = [f for f in dir_path.iterdir() if f.is_file() and date_pattern.search(f.name)]
        if not files:
            logger.warning(f"No JSON files found in {dir_path}")
            return pd.DataFrame()

        all_data = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.parse_file, file): file for file in files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel parsing"):
                file = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        all_data.append(df)
                except Exception as e:
                    logger.error(f"Error processing {file}: {e}")

        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            return df_all
        else:
            return pd.DataFrame()

    def parse_stream(self, stream: Generator[str, None, None]) -> pd.DataFrame:
        """Parse log events from a stream (line by line) for real-time processing"""
        events = []
        for line in stream:
            try:
                json_event = json.loads(line.strip())
                if parsed := self._process_event(json_event):
                    events.append(parsed)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in stream: {line[:100]}...")
        return pd.DataFrame(events)

    
    def dict_to_json(self, val):
        if isinstance(val, dict):
            return json.dumps(val)
        else:
            return str(val) if val is not None else ""

    def export_results(self, df: pd.DataFrame, name: str = "cowrie_events"):
        if df.empty:
            logger.warning("No data to export")
            return
        
        if 'mitre_ttp' in df.columns:
            df['mitre_ttp'] = df['mitre_ttp'].apply(self.dict_to_json)

        NUMERIC_COLUMNS = ['dst_port', 'src_port', 'connection_duration']
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = df[col].replace(r'^\s*$', np.nan, regex=True).infer_objects()
                df[col] = pd.to_numeric(df[col], errors='coerce')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Remove only the final extension, preserve entire stem (multi-dot!)
        if '.' in name:
            base_name = name.rsplit('.', 1)[0]
        else:
            base_name = name
        base_name_clean = base_name.replace('.', '_')
        out_name = f"{base_name_clean}_processed_{timestamp}.{self.output_format}"

        output_path = self.output_dir / out_name
        print(f"Exporting to: {output_path.resolve()}")

        # Export in chosen format
        if self.output_format == "parquet":
            df.to_parquet(output_path)
        elif self.output_format == "csv":
            df.to_csv(output_path, index=False)
        elif self.output_format in ("json", "jsonl", "jsonlines"):
            df.to_json(output_path, orient="records", lines=True)
        else:
            logger.error(f"Unsupported output format: {self.output_format}")
            return

        logger.info(f"Exported {len(df)} events to {output_path}")

# Usage Example
if __name__ == "__main__":
    args = parse_args()
    # State logs
    local_state_log_path = Path(args.output_dir) / LOCAL_STATE_LOG
    remote_state_log_path = Path(args.output_dir) / REMOTE_STATE_LOG
    if args.force and local_state_log_path.exists():
        print("Forcing reprocessing; removing local state log...")
        local_state_log_path.unlink()
    if args.force and remote_state_log_path.exists():
        print("Forcing reprocessing; removing remote state log...")
        remote_state_log_path.unlink()
    state = load_state(local_state_log_path)
    remote_state = load_state(remote_state_log_path)
    session_ids = args.session_ids.split(",") if args.session_ids else None
    event_ids = args.event_ids.split(",") if args.event_ids else None
    date_range = parse_date_range(args.date)
    processor = CowrieJSONProcessor(
        output_dir=args.output_dir,
        output_format=args.output_format,
        max_workers=args.max_workers,
        geo_enabled=not args.no_geo,
    )
    if is_url(args.input_path):
        print(f"Streaming remote log: {args.input_path}")
        last_state = remote_state.get(args.input_path, {})
        last_line_hash = last_state.get("last_line_hash", None)
        gen = stream_remote_log_incremental(args.input_path, last_line_hash=last_line_hash)
        events = []
        hash_of_last = None
        for line in gen:
            hash_of_last = line_hash(line)
            try:
                json_event = json.loads(line.strip())
                evt = processor._process_event(json_event)
                if evt:
                    events.append(evt)
            except Exception:
                logger.warning(f"Invalid JSON in remote log: {line[:100]}...")
        if events:
            df = pd.DataFrame(events)
            df = filter_dataframe(df, date_range=date_range, session_ids=session_ids, event_ids=event_ids)
            processor.export_results(df, name=args.input_path.split("/")[-1])
            remote_state[args.input_path] = {"last_line_hash": hash_of_last}
            save_state(remote_state_log_path, remote_state)
            print("Completed remote log streaming and export.")
        else:
            print("No new valid events found from remote log.")
    else:
        input_path = Path(args.input_path)
        files_to_process = []
        if input_path.is_file():
            files_to_process = [input_path]
        elif input_path.is_dir():
            files_to_process = sorted(
                f for f in input_path.iterdir()
                if f.is_file() and (".json" in f.name or ".log" in f.name)
            )
        else:
            print(f"Input path {input_path} is neither a valid file nor directory.")
            exit(1)
        out_dfs = []
        for file_path in files_to_process:
            key = str(file_path.resolve())
            mtime = os.path.getmtime(file_path)
            state_entry = state.get(key, {})
            already = state_entry.get('lines', 0)
            last_mod = state_entry.get('mtime', 0)
            to_skip = already if last_mod == mtime else 0
            to_skip = max(to_skip, args.skip)
            total_lines = sum(1 for _ in open(file_path, encoding='utf-8'))
            if to_skip >= total_lines:
                print(f"Already parsed: {file_path}")
                continue
            print(f"Parsing {file_path.name} (skip {to_skip}, limit {args.limit or 'ALL'}) ...")
            events = []
            with open(file_path, 'r', encoding='utf-8') as f, tqdm(total=total_lines-to_skip, desc=f"Reading {file_path.name}") as pbar:
                for i, line in enumerate(f):
                    if i < to_skip: continue
                    if args.limit is not None and (i - to_skip) >= args.limit: break
                    try:
                        json_evt = json.loads(line.strip())
                        evt = processor._process_event(json_evt)
                        if evt:
                            events.append(evt)
                    except Exception:
                        continue
                    pbar.update(1)
            df = pd.DataFrame(events)
            if not df.empty:
                df = filter_dataframe(df, date_range=date_range, session_ids=session_ids, event_ids=event_ids)
                processor.export_results(df, name=file_path.name)
                out_dfs.append(df)
                state[key] = {"lines": i + 1, "mtime": mtime}
                save_state(local_state_log_path, state)
            else:
                print(f"No valid events found in {file_path}")
        if not out_dfs:
            print("No new data parsed.")
        else:
            print(f"Completed parsing {len(out_dfs)} file(s).")
