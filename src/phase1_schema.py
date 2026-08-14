"""Verified Edge-IIoTset schema and feature-role policy for Phase 1 only."""

from __future__ import annotations

EXPECTED_COLUMNS = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.dst.proto_ipv4", "arp.opcode",
    "arp.hw.size", "arp.src.proto_ipv4", "icmp.checksum", "icmp.seq_le",
    "icmp.transmit_timestamp", "icmp.unused", "http.file_data", "http.content_length",
    "http.request.uri.query", "http.request.method", "http.referer", "http.request.full_uri",
    "http.request.version", "http.response", "http.tls_port", "tcp.ack", "tcp.ack_raw",
    "tcp.checksum", "tcp.connection.fin", "tcp.connection.rst", "tcp.connection.syn",
    "tcp.connection.synack", "tcp.dstport", "tcp.flags", "tcp.flags.ack", "tcp.len",
    "tcp.options", "tcp.payload", "tcp.seq", "tcp.srcport", "udp.port", "udp.stream",
    "udp.time_delta", "dns.qry.name", "dns.qry.name.len", "dns.qry.qu", "dns.qry.type",
    "dns.retransmission", "dns.retransmit_request", "dns.retransmit_request_in",
    "mqtt.conack.flags", "mqtt.conflag.cleansess", "mqtt.conflags", "mqtt.hdrflags",
    "mqtt.len", "mqtt.msg_decoded_as", "mqtt.msg", "mqtt.msgtype", "mqtt.proto_len",
    "mqtt.protoname", "mqtt.topic", "mqtt.topic_len", "mqtt.ver", "mbtcp.len",
    "mbtcp.trans_id", "mbtcp.unit_id", "Attack_label", "Attack_type",
]

NODE_ID = {"ip.src_host", "ip.dst_host"}
TEMPORAL = {"frame.time"}
LABEL = {"Attack_label", "Attack_type"}

# Packet bodies, queries, sequence/checksum fields, and additional endpoint IDs are
# deliberately excluded: they are identifiers/high-cardinality content, not traffic state.
DROP = {
    "arp.dst.proto_ipv4", "arp.src.proto_ipv4", "icmp.checksum", "icmp.seq_le",
    "icmp.transmit_timestamp", "http.file_data", "http.request.uri.query", "http.referer",
    "http.request.full_uri", "tcp.ack_raw", "tcp.checksum", "tcp.options", "tcp.payload",
    "tcp.seq", "dns.qry.name", "mqtt.msg", "mqtt.topic", "mbtcp.trans_id",
}
CATEGORICAL = {
    "http.request.method", "http.request.version", "http.response", "mqtt.msg_decoded_as",
    "mqtt.protoname",
}
EDGE_FEATURE = {
    "arp.opcode", "arp.hw.size", "icmp.unused", "http.content_length", "http.tls_port",
    "tcp.ack", "tcp.connection.fin", "tcp.connection.rst", "tcp.connection.syn",
    "tcp.connection.synack", "tcp.dstport", "tcp.flags", "tcp.flags.ack", "tcp.len",
    "tcp.srcport", "udp.port", "udp.stream", "udp.time_delta", "dns.qry.name.len",
    "dns.qry.qu", "dns.qry.type", "dns.retransmission", "dns.retransmit_request",
    "dns.retransmit_request_in", "mqtt.conack.flags", "mqtt.conflag.cleansess",
    "mqtt.conflags", "mqtt.hdrflags", "mqtt.len", "mqtt.msgtype", "mqtt.proto_len",
    "mqtt.topic_len", "mqtt.ver", "mbtcp.len", "mbtcp.unit_id",
}


def feature_mapping() -> dict[str, str]:
    """Return exactly one Phase-1 role for every verified source column."""
    mapping: dict[str, str] = {}
    for column in EXPECTED_COLUMNS:
        if column in NODE_ID:
            mapping[column] = "NODE_ID"
        elif column in TEMPORAL:
            mapping[column] = "TEMPORAL"
        elif column in LABEL:
            mapping[column] = "LABEL"
        elif column in DROP:
            mapping[column] = "DROP"
        elif column in CATEGORICAL:
            mapping[column] = "CATEGORICAL"
        elif column in EDGE_FEATURE:
            mapping[column] = "EDGE_FEATURE"
        else:
            raise RuntimeError(f"No feature role assigned for {column}")
    return mapping
