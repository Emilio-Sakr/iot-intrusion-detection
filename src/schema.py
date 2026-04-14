# src/schema.py
from pydantic import BaseModel
from typing import Literal

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
    "radius", "covariance", "variance", "weight"
]

ATTACK_CLASSES = [
    "BenignTraffic",
    "DDoS-ICMP_Flood",
    "DDoS-UDP_Flood",
    "DDoS-TCP_SYN_Flood",
    "DoS-UDP_Flood",
    "Mirai-greeth_flood",
    "MITM-ArpSpoofing",
    "DNS_Spoofing",
    "Recon-PortScan",
    "BrowserHijacking",
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