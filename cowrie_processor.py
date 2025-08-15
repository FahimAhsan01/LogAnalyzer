import os
import re
import json
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedCowrieParser:
    """Advanced parser for Cowrie logs with MITRE ATT&CK categorization"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MITRE ATT&CK patterns
        self.attack_patterns = {
            'Discovery': [
                # T1033: System Owner/User Discovery
                'whoami', 'id', 'w', 'cat /etc/passwd', 'cat /etc/shadow',
                # T1018: Remote Services Discovery
                'netstat', 'ss', 'nmap', 'lsof', 'sockstat', 'cat /proc/net/tcp',
                # T1082: System Information Discovery
                'uname', 'lsb_release', 'cat /proc/version', 'hostnamectl', 'cat /etc/os-release',
                # T1049: System Network Connections Discovery
                'ifconfig', 'ip a', 'route', 'traceroute', 'tcpdump', 'ping', 'telnet',
                # T1016: System Network Configuration Discovery
                'cat /etc/network/interfaces', 'cat /etc/sysctl.conf', 'iptables-save', 'ipset list'
            ],
            'Persistence': [
                # T1053: Scheduled Task/Job
                'crontab', 'at', 'anacrontab', 'systemctl enable', 'systemctl start',
                # T1504: Command Staging
                'wget', 'curl', 'python -c "import os; os.system"', 'base64 -d',
                # T1543: Create or Modify System Process
                'systemd-timer', 'systemd-unit', 'init.d', 'rc.local', 'motd',
                # T1546: Event Triggered Execution
                'inotifywait', 'incrontab', 'tmpwatch', 'logrotate', 'cron.d/'
            ],
            'Execution': [
                # T1059: Command and Scripting Interpreter
                'bash', 'sh', 'python', 'perl', 'ruby', 'php', 'awk', 'sed',
                # T1504: Command Staging
                './', 'sh -c', 'eval', 'exec', 'xargs', 'find . -exec',
                # T1204: User Execution
                'xdg-open', 'xterm', 'gnome-terminal', 'konsole', 'termux'
            ],
            'Credential_Access': [
                # T1003: OS Credential Dumping
                'cat /etc/shadow', 'sudo -l', 'cat ~/.ssh/authorized_keys', 'getent passwd',
                # T1006: Direct Command Execution
                'su', 'sudo', 'runuser', 'kadmin', 'smbclient',
                # T1021: Remote Services
                'ssh-keygen', 'ssh-copy-id', 'expect', 'plink', 'rsh'
            ],
            'Lateral_Movement': [
                # T1021: Remote Services
                'scp', 'ssh', 'rsync', 'nc', 'netcat', 'rsh', 'telnet', 'ftp',
                # T1018: Remote Services Discovery
                'ssh -J', 'ssh -L', 'ssh -R', 'ssh -D', 'ssh -fN',
                # T1548: Abuse Elevation Control Mechanism
                'sudoedit', 'pkexec', 'polkit', 'dbind', 'mount'
            ]
        }
        
        # Regex patterns with proper syntax
        self.timestamp_pattern = re.compile(
            r'(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\.\d+Z?)'
        )
        self.login_pattern = re.compile(
            r'login attempt \[(.*?)/(.*?)\] (succeeded|failed)'
        )
        self.command_pattern = re.compile(
            r'(?:command input:\s*|CMD:\s*|\[CMD\]\s+)(.*?)(?=\\n|;|&|\||$)',
            re.IGNORECASE | re.DOTALL
        )
        self.connection_pattern = re.compile(
            r'New connection: ([\d\.]+):\d+ \(([\d\.]+):(\d+)\) \[session: (\w+)\]'
        )

    def parse_directory(self, path_str: str):
        """Process all log files in a directory"""
        path_obj = Path(path_str)
        if not path_obj.exists():
            raise FileNotFoundError(f"Directory not found: {path_str}")
            
        for child in path_obj.iterdir():
            if child.is_file():
                try:
                    self.parse_file(str(child))
                except PermissionError:
                    logger.error(f"Permission denied for file: {child}")
                except Exception as e:
                    logger.error(f"Error processing {child}: {e}")

    def parse_file(self, file_path: str):
        """Main parsing method with progress tracking"""
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        total_lines = self._count_lines(path_obj)
        return self._process_file(path_obj, total_lines)

    def _count_lines(self, path_obj: Path) -> int:
        """Count lines for progress bar"""
        try:
            with open(path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.error(f"Error counting lines: {e}")
            return 0

    def _process_file(self, path_obj: Path, total_lines: int) -> pd.DataFrame:
        """Process file with format detection"""
        entries = []
        try:
            with open(path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                with tqdm(total=total_lines, desc=f"Processing {path_obj.name}", unit="line") as pbar:
                    for line in f:
                        entry = self._parse_line(line.strip(), path_obj.name)
                        if entry:
                            entries.append(entry)
                        pbar.update(1)
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in {path_obj}: {e}")
        
        return self._post_process(entries, path_obj)

    def _parse_line(self, line: str, filename: str) -> Optional[Dict]:
        """Parse individual log line"""
        if not line:
            return None

        # Try JSON format first
        try:
            return self._parse_json(line, filename)
        except json.JSONDecodeError:
            return self._parse_text(line, filename)
        except Exception as e:
            logger.debug(f"JSON parse error: {e}")
            return None

    def _parse_json(self, line: str, filename: str) -> Optional[Dict]:
        """Parse JSON log entry"""
        entry = json.loads(line)
        event_type = entry.get('eventid', 'unknown')
        
        base = {
            'timestamp': entry.get('timestamp'),
            'event_type': event_type,
            'src_ip': entry.get('src_ip', ''),
            'session': entry.get('session', ''),
            'source_file': filename,
            'raw': line
        }

        processors = {
            'cowrie.session.connect': self._process_connection,
            'cowrie.login.success': self._process_login,
            'cowrie.login.failed': self._process_login,
            'cowrie.command.input': self._process_command
        }

        if processor := processors.get(event_type):
            return {**base, **processor(entry)}
        return None

    def _parse_text(self, line: str, filename: str) -> Optional[Dict]:
        """Parse text log line with enhanced handling"""
        timestamp_match = self.timestamp_pattern.match(line)
        if not timestamp_match:
            return None
           
        entry = {
            'timestamp': timestamp_match.group('timestamp'),
            'source_file': filename,
            'raw': line
        }

        # Decode byte strings in text log entries
        decoded_line = self._decode_bytes(line)
        
        if 'login attempt' in decoded_line:
            return {**entry, **self._process_text_login(decoded_line)}
        elif any(pattern in decoded_line for pattern in ['command input:', 'CMD:', '[CMD]']):
            return {**entry, **self._process_text_command(decoded_line)}
        elif 'New connection:' in decoded_line:
            return {**entry, **self._process_text_connection(decoded_line)}
        elif 'Command found:' in decoded_line:
            return {**entry, **self._process_text_command(decoded_line)}
        
        return None

    def _decode_bytes(self, line: str) -> str:
        """Decode byte strings in log entries"""
        return line.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

    def _clean_field(self, field: str) -> str:
        """Comprehensive cleaning of artifacts"""
        cleaned = re.sub(r'\\x[0-9a-f]{2}', '', field)
        cleaned = re.sub(r'\\[btnfr"\']', lambda m: eval(f'"{m.group()}"'), cleaned)
        return cleaned.strip("b'").strip('"').strip()

    def _process_connection(self, entry: Dict) -> Dict:
        """Process connection events"""
        return {
            'src_ip': entry.get('src_ip', ''),
            'src_port': entry.get('src_port', 0),
            'dst_port': entry.get('dst_port', 0),
            'event_category': 'connection'
        }

    def _process_login(self, entry: Dict) -> Dict:
        """Process login events with byte decoding"""
        return {
            'username': self._clean_field(entry.get('username', '')),
            'password': self._clean_field(entry.get('password', '')),
            'success': entry.get('eventid') == 'cowrie.login.success',
            'event_category': 'authentication'
        }

    def _process_command(self, entry: Dict) -> Dict:
        """Process command events with enhanced categorization"""
        command = entry.get('input', '')
        return {
            'command': command,
            'attack_category': self._categorize_command(command),
            'event_category': 'command'
        }

    def _process_text_login(self, line: str) -> Dict:
        """Extract login details from text logs with decoding"""
        match = self.login_pattern.search(line)
        if match:
            return {
                'username': self._clean_field(match.group(1)),
                'password': self._clean_field(match.group(2)),
                'success': match.group(3) == 'succeeded',
                'event_category': 'authentication'
            }
        return {}

    def _process_text_command(self, line: str) -> Dict:
        # Handle "Command found: uname -s -m" format
        found_match = re.search(r'Command found:\s+(.+)', line)
        if found_match:
            command = found_match.group(1).strip()
            return {
                'command': command,
                'attack_category': self._categorize_command(command),
                'event_category': 'command'
            }
        return {}

    def _process_text_connection(self, line: str) -> Dict:
        """Extract connection details from text logs"""
        match = self.connection_pattern.search(line)
        if match:
            return {
                'src_ip': match.group(1),
                'dst_ip': match.group(2),
                'dst_port': match.group(3),
                'session': match.group(4),
                'event_category': 'connection'
            }
        return {}

    def _categorize_command(self, command: str) -> str:
        """Enhanced MITRE ATT&CK categorization"""
        command = command.lower()
        for category, patterns in self.attack_patterns.items():
            if any(p.lower() in command for p in patterns):
                return category
        return 'Other'

    def _post_process(self, entries: List[Dict], path_obj: Path) -> pd.DataFrame:
        """Save results and generate RAG input"""
        if not entries:
            logger.warning(f"No valid events in {path_obj.name}")
            return pd.DataFrame()

        df = pd.DataFrame(entries)
        df['session'] = df['session'].ffill()
        session_commands = df.groupby('session')['command'].apply(
            lambda x: '\n'.join(x.dropna())
        )
        # Save parsed data
        output_file = self.output_dir / f"{path_obj.stem}_parsed.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} events to {output_file}")

        # Generate RAG input
        if 'command' in df.columns:
            self._generate_rag_input(df, path_obj)
        

        
        return df

    def _generate_rag_input(self, df: pd.DataFrame, path_obj: Path):
        # Get all non-empty commands
        commands = df[df['command'].notna()]['command'].tolist()
        
        # Split compound commands (; & |)
        split_commands = []
        for cmd in commands:
            split_commands.extend(re.split(r'[;&|]', cmd))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_commands = [x.strip() for x in split_commands 
                        if (x.strip() not in seen and not seen.add(x.strip()))]
        
        # Write to file
        rag_file = self.output_dir / f"{path_obj.stem}_commands.txt"
        with open(rag_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(unique_commands))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python cowrie_processor.py <path/to/logs>")
        sys.exit(1)
    
    parser = EnhancedCowrieParser()
    input_path = sys.argv[1]
    
    if os.path.isdir(input_path):
        parser.parse_directory(input_path)
    elif os.path.isfile(input_path):
        parser.parse_file(input_path)
    else:
        logger.error(f"Invalid path: {input_path}")
        sys.exit(1)
