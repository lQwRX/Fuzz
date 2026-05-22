# capture/message_collector.py
import os
import random
import struct
from typing import List
from utils.file_utils import ensure_dir, save_text, load_text
from config import PCAP_PATHS, AUTO_GENERATE_SYNTHETIC, SYNTHETIC_NUM_SAMPLES, SYNTHETIC_MAX_LEN, RAW_MSG_FILE

# 如果存在 scapy 则使用，否则仅用模拟生成
try:
    from scapy.all import rdpcap, TCP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: scapy not installed, cannot read pcap. Will use synthetic data only.")

class MessageCollector:
    @staticmethod
    def extract_from_single_pcap(pcap_path: str, port: int = 502) -> List[str]:
        if not SCAPY_AVAILABLE or not os.path.exists(pcap_path):
            return []
        packets = rdpcap(pcap_path)
        messages = []
        for pkt in packets:
            if TCP in pkt and pkt[TCP].dport == port:
                payload = bytes(pkt[TCP].payload)
                if len(payload) >= 7:
                    messages.append(payload.hex())
        return messages

    @staticmethod
    def extract_from_multiple_pcaps(pcap_paths: List[str], port: int = 502) -> List[str]:
        all_msgs = []
        for path in pcap_paths:
            msgs = MessageCollector.extract_from_single_pcap(path, port)
            print(f"  Extracted {len(msgs)} messages from {path}")
            all_msgs.extend(msgs)
        return all_msgs

    @staticmethod
    def generate_synthetic_modbus(num_samples: int, max_len: int) -> List[str]:
        """生成模拟 Modbus 请求（不均匀功能码分布）"""
        def random_modbus_request():
            # 功能码权重：03读保持寄存器占50%
            func_codes = [1,2,3,4,5,6,15,16]
            weights = [0.1,0.05,0.5,0.05,0.05,0.05,0.1,0.1]
            func_code = random.choices(func_codes, weights=weights)[0]
            trans_id = random.randint(0, 65535)
            proto_id = 0
            unit_id = 1
            address = random.randint(0, 100)

            if func_code in [1,2,3,4]:
                quantity = random.randint(1, 100)
                pdu = struct.pack('>B H H', func_code, address, quantity)
            elif func_code in [5,6]:
                value = random.randint(0, 65535)
                pdu = struct.pack('>B H H', func_code, address, value)
            elif func_code == 15:
                quantity = random.randint(1, 100)
                byte_count = (quantity + 7) // 8
                data = bytes([random.getrandbits(8) for _ in range(byte_count)])
                pdu = struct.pack('>B H H B', func_code, address, quantity, byte_count) + data
            elif func_code == 16:
                quantity = random.randint(1, 100)
                byte_count = quantity * 2
                data = struct.pack('>' + 'H'*quantity, *[random.randint(0,65535) for _ in range(quantity)])
                pdu = struct.pack('>B H H B', func_code, address, quantity, byte_count) + data
            else:
                pdu = struct.pack('>B', func_code)

            length = len(pdu) + 1
            header = struct.pack('>H H H B', trans_id, proto_id, length, unit_id)
            msg = header + pdu
            if len(msg) > max_len:
                return None
            return msg.hex()

        messages = []
        while len(messages) < num_samples:
            msg = random_modbus_request()
            if msg:
                messages.append(msg)
        return messages

    @staticmethod
    def collect(output_file: str = RAW_MSG_FILE) -> List[str]:
        ensure_dir(os.path.dirname(output_file))
        msgs = []
        if PCAP_PATHS and SCAPY_AVAILABLE:
            print("Extracting from pcap files...")
            msgs = MessageCollector.extract_from_multiple_pcaps(PCAP_PATHS)
        if not msgs and AUTO_GENERATE_SYNTHETIC:
            print(f"Generating {SYNTHETIC_NUM_SAMPLES} synthetic Modbus messages...")
            msgs = MessageCollector.generate_synthetic_modbus(SYNTHETIC_NUM_SAMPLES, SYNTHETIC_MAX_LEN)
        if not msgs:
            raise RuntimeError("No data collected. Provide pcap files or enable synthetic generation.")
        save_text(msgs, output_file)
        print(f"Saved {len(msgs)} raw messages to {output_file}")
        return msgs

if __name__ == "__main__":
    # 测试运行
    msgs = MessageCollector.collect()
    print("First 3 messages:")
    for m in msgs[:3]:
        print(m)