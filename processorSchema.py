class CowrieJSONProcessor:
    SCHEMA = [
        'timestamp', 'event_type', 'session', 'src_ip', 'sensor', 'username', 'password',
        'login_success', 'command', 'mitre_ttp', 'anomaly_flag', 'connection_duration',
        'protocol', 'dst_ip', 'dst_port', 'src_port', 'url', 'outfile', 'sha256', 'size',
        'hassh', 'kex_algs', 'enc_algs', 'country', 'region', 'city', 'org', 'asn',
        'location',
        # any other fields you enrich, plus per-row fields like 'page_content'
    ]
    ...
    def process_event(self, event):
        processed = {field: None for field in self.SCHEMA}  # start with all fields set to None
        ... # your current processing logic, writing values
        return processed
