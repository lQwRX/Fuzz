# preprocessing/numericalizer.py
from utils.conversion_utils import hex_to_bytes

class Numericalizer:
    @staticmethod
    def messages_to_sequences(messages_hex, vocab):
        """返回整数序列列表"""
        sequences = []
        for msg in messages_hex:
            byte_data = hex_to_bytes(msg)
            seq = [vocab[b] for b in byte_data]
            sequences.append(seq)
        return sequences