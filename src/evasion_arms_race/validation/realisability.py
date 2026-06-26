"""pcap-level realisability validation (Layer A, todo item 7).

Take a generated adversarial feature vector and verify it corresponds to legal,
sendable traffic. This is where the 'unreconstructable' aggregate features
(packet-length stats under CICFlowMeter) must be resolved against real packets.
"""
