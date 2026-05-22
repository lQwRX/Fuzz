# preprocessing/padder.py
import numpy as np

class Padder:
    @staticmethod
    def pad_sequences(sequences, max_len, pad_val=1):
        padded = np.full((len(sequences), max_len), pad_val, dtype=np.int64)
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_len)
            padded[i, :length] = seq[:length]
        return padded

    @staticmethod
    def truncate_sequences(sequences, max_len):
        return [seq[:max_len] for seq in sequences]