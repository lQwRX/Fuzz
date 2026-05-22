import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_MSG_FILE, DATA_DIR, MIN_MSG_LEN, MAX_MSG_LEN


def load_messages(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []
    with open(filepath, 'r') as f:
        return [l.strip() for l in f if l.strip()]


def clean(messages):
    cleaned = []
    seen = set()
    for msg in messages:
        byte_len = len(msg) // 2
        if MIN_MSG_LEN <= byte_len <= MAX_MSG_LEN and msg not in seen:
            cleaned.append(msg)
            seen.add(msg)
    return cleaned


def build_vocab(messages):
    byte_set = set()
    for msg in messages:
        for b in bytes.fromhex(msg):
            byte_set.add(b)
    sorted_bytes = sorted(byte_set)
    vocab = {b: i + 2 for i, b in enumerate(sorted_bytes)}
    vocab['<PAD>'] = 1
    return vocab


def numericalize(messages, vocab):
    sequences = []
    for msg in messages:
        seq = [vocab[b] for b in bytes.fromhex(msg)]
        sequences.append(seq)
    return sequences


def pad_sequences(sequences, max_len, pad_val=1):
    padded = np.full((len(sequences), max_len), pad_val, dtype=np.int64)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    return padded


def main():
    print("=" * 60)
    print("  Step 1: Data Preprocessing")
    print("=" * 60)

    raw = load_messages(RAW_MSG_FILE)
    print(f"Loaded {len(raw)} raw messages from {RAW_MSG_FILE}")

    cleaned = clean(raw)
    print(f"After clean (length filter + dedup): {len(cleaned)}")

    vocab = build_vocab(cleaned)
    pad_idx = vocab['<PAD>']
    print(f"Vocab size: {len(vocab)}, PAD idx: {pad_idx}")

    sequences = numericalize(cleaned, vocab)
    max_len = max(len(s) for s in sequences)
    max_len = min(max_len, MAX_MSG_LEN)
    print(f"Max sequence length: {max_len}")

    truncated = [s[:max_len] for s in sequences]
    data = pad_sequences(truncated, max_len, pad_idx)
    print(f"Data matrix shape: {data.shape}")

    np.save(os.path.join(DATA_DIR, "data.npy"), data)
    with open(os.path.join(DATA_DIR, "vocab.json"), 'w') as f:
        json.dump({str(k): v for k, v in vocab.items()}, f, indent=2)
    with open(os.path.join(DATA_DIR, "max_len.txt"), 'w') as f:
        f.write(str(max_len))

    print(f"Saved to {DATA_DIR}/")
    print(f"  data.npy  shape={data.shape}")
    print(f"  vocab.json  ({len(vocab)} entries)")
    print(f"  max_len.txt  ({max_len})")
    return data.shape[0]


if __name__ == "__main__":
    main()
