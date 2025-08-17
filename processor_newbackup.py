import os
import json
import logging
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Union, Generator
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv(find_dotenv())
IPINFO_TOKEN = os.environ.get("IPINFO_TOKEN")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CowrieJSONProcessor:
    """Optimized parser for Cowrie JSON logs with session reconstruction and enhanced features"""

    def __init__(self, output_dir: str = "processed_data", output_format: str = "parquet", max_workers: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.geo_cache = {}
        self.session_cache = {}  # Track sessions across events
        self.attack_matrix = self._load_mitre_matrix()
        self.output_format = output_format.lower()
        self.max_workers = max_workers
        self.anomaly_commands = [
            # Sample anomaly patterns to flag suspicious commands
            'rm -rf /', 'wget http', 'curl http', 'nc -l', 'netcat', 'exec ', 'base64 -d', 'chmod 777',
            'ssh-keyscan', 'ssh ', 'scp ', 'telnet ', 'ftp ', 'sftp ', 'python -c', 'perl -e', 'eval '
        ]

    def _load_mitre_matrix(self) -> Dict[str, List[str]]:
        """Expanded MITRE ATT&CK patterns with technique IDs including wider coverage"""
        # This matrix is extended with additional well-known attack techniques and keywords
        return {
            'T1087': ['whoami', 'id', 'w', 'getent passwd', 'cat /etc/passwd', 'finger', 'id -u'],
            'T1059': ['bash', 'sh', 'python', 'perl', 'php', 'awk', r'\./', 'powershell', 'cmd.exe', 'cscript'],
            'T1552': ['unshadow', 'cat /etc/shadow', 'find / -name id_rsa', 'ssh-keygen', 'ssh-add'],
            'T1021': ['ssh ', 'scp ', 'telnet ', 'ftp ', 'sftp ', 'rlogin', 'rexec'],
            'T1070': ['rm -rf', 'shred', 'echo "" > ', 'logrotate --force', 'wipe', 'dd if=', 'cat /dev/null >'],
            'T1056': ['keylogger', 'strace', 'ltrace', 'cat .ssh/known_hosts', 'xinput', 'wireshark'],
            'T1569': ['systemctl', 'service', '/etc/init.d/', 'killall', 'pkill', 'cron', 'at '],
            'T1003': ['hashdump', 'mimikatz', 'lsass', 'procdump'],  # Credential dumping
            'T1135': ['net view', 'netsh', 'netstat', 'arp -a'],   # Network sniffing
            'T1218': ['regsvr32', 'mshta', 'rundll32'],            # Signed binary proxy execution
            'T1105': ['curl', 'wget', 'scp'],                      # Ingress tool transfer
        }

    def _geolocate_ip(self, ip: str) -> Dict[str, str]:
        """Cached IP geolocation with error handling"""
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
        """Flag suspicious commands based on predefined patterns"""
        command_lower = command.lower()
        return any(anom in command_lower for anom in self.anomaly_commands)

    def _process_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Core event processing with session awareness, enriched metadata, and anomaly detection"""
        processed = {
            'timestamp': datetime.fromisoformat(event['timestamp'].rstrip('Z')),
            'event_type': event['eventid'],
            'session': event.get('session', ''),
            'src_ip': event.get('src_ip', ''),
            'sensor': event.get('sensor', '')
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

        elif event_type in ('cowrie.login.success', 'cowrie.login.failed'):
            creds = {
                'username': event.get('username', ''),
                'password': event.get('password', ''),
                'success': (event_type == 'cowrie.login.success'),
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
            mitre_ttps = self._categorize_ttp(command)
            anomaly_flag = self._detect_anomaly(command)
            processed.update({
                'command': command,
                'mitre_ttp': mitre_ttps,
                'anomaly_flag': anomaly_flag
            })

        elif event_type == 'cowrie.session.file_download':
            processed.update({
                'url': event.get('url', ''),
                'outfile': event.get('outfile', ''),
                'sha256': event.get('sha256', ''),
                'size': event.get('size', 0)
            })

        elif event_type == 'cowrie.client.kex':
            processed.update({
                'hassh': event.get('hassh', ''),
                'kex_algs': event.get('kexAlgs', []),
                'enc_algs': event.get('encCS', [])
            })

        # Propagate latest login info and session metadata
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

        files = list(dir_path.glob("*.json"))
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

    def export_results(self, df: pd.DataFrame, name: str = "cowrie_events"):
        """Save parsed data in the specified format with timestamp"""
        if df.empty:
            logger.warning("No data to export")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.output_format == "parquet":
            output_path = self.output_dir / f"{name}_{timestamp}.parquet"
            df.to_parquet(output_path)
        elif self.output_format == "csv":
            output_path = self.output_dir / f"{name}_{timestamp}.csv"
            df.to_csv(output_path, index=False)
        elif self.output_format in ("json", "jsonl", "jsonlines"):
            output_path = self.output_dir / f"{name}_{timestamp}.jsonl"
            df.to_json(output_path, orient="records", lines=True)
        else:
            logger.error(f"Unsupported output format: {self.output_format}")
            return

        logger.info(f"Exported {len(df)} events to {output_path}")


# Usage Example
if __name__ == "__main__":
    import sys

    output_format = "parquet"
    processor = CowrieJSONProcessor(output_format=output_format)

    if len(sys.argv) < 2:
        print("Usage: python processor_new_enhanced.py <path/to/cowrie.json or directory> [output_format]")
        print("Output formats supported: parquet (default), csv, jsonl")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        processor.output_format = sys.argv[2].lower()

    if log_path.is_dir():
        df = processor.parse_files_parallel(log_path)
    else:
        df = processor.parse_file(log_path)

    if not df.empty:
        processor.export_results(df)
        print(f"Processed {len(df)} events from {log_path}")
    else:
        print("No valid events found")
