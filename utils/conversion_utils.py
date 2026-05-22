# utils/conversion_utils.py
def hex_to_bytes(hex_str: str) -> bytes:
    return bytes.fromhex(hex_str)

def bytes_to_hex(byte_data: bytes) -> str:
    return byte_data.hex()

def bytes_to_int_seq(byte_data: bytes) -> list:
    return list(byte_data)

def int_seq_to_hex(int_seq: list) -> str:
    return bytes(int_seq).hex()