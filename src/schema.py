# src/schema.py
from pydantic import BaseModel

LABEL_COLUMN = "label"

FEATURE_COLUMNS = [
    "flow_duration", "header_length", "protocol_type",
    "duration", "rate", "srate", "drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "ack_count", "syn_count",
    "fin_count", "urg_count", "rst_count",
    "http", "https", "dns", "telnet", "smtp",
    "ssh", "irc", "tcp", "udp", "dhcp",
    "arp", "icmp", "igmp", "ipv", "llc",
    "tot_sum", "min", "max", "avg", "std",
    "tot_size", "iat", "number", "magnitue",
    "radius", "covariance", "variance", "weight",
]

# Full 34-class taxonomy present in data/processed/sample.parquet.
# Grouped by family for readability; LabelEncoder will sort internally.
ATTACK_CLASSES = [
    "BenignTraffic",
    # DDoS family
    "DDoS-ACK_Fragmentation",
    "DDoS-HTTP_Flood",
    "DDoS-ICMP_Flood",
    "DDoS-ICMP_Fragmentation",
    "DDoS-PSHACK_Flood",
    "DDoS-RSTFINFlood",
    "DDoS-SYN_Flood",
    "DDoS-SlowLoris",
    "DDoS-SynonymousIP_Flood",
    "DDoS-TCP_Flood",
    "DDoS-UDP_Flood",
    "DDoS-UDP_Fragmentation",
    # DoS family
    "DoS-HTTP_Flood",
    "DoS-SYN_Flood",
    "DoS-TCP_Flood",
    "DoS-UDP_Flood",
    # Mirai family
    "Mirai-greeth_flood",
    "Mirai-greip_flood",
    "Mirai-udpplain",
    # Recon family
    "Recon-HostDiscovery",
    "Recon-OSScan",
    "Recon-PingSweep",
    "Recon-PortScan",
    # Spoofing / MITM
    "DNS_Spoofing",
    "MITM-ArpSpoofing",
    # Scanning / brute
    "DictionaryBruteForce",
    "VulnerabilityScan",
    # Web / application layer
    "Backdoor_Malware",
    "BrowserHijacking",
    "CommandInjection",
    "SqlInjection",
    "Uploading_Attack",
    "XSS",
]


class TrafficRecord(BaseModel):
    flow_duration: float
    header_length: float
    protocol_type: float
    duration: float
    rate: float
    srate: float
    drate: float
    fin_flag_number: float
    syn_flag_number: float
    rst_flag_number: float
    psh_flag_number: float
    ack_flag_number: float
    ece_flag_number: float
    cwr_flag_number: float
    ack_count: float
    syn_count: float
    fin_count: float
    urg_count: float
    rst_count: float
    http: float
    https: float
    dns: float
    telnet: float
    smtp: float
    ssh: float
    irc: float
    tcp: float
    udp: float
    dhcp: float
    arp: float
    icmp: float
    igmp: float
    ipv: float
    llc: float
    tot_sum: float
    min: float
    max: float
    avg: float
    std: float
    tot_size: float
    iat: float
    number: float
    magnitue: float
    radius: float
    covariance: float
    variance: float
    weight: float


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
