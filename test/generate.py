import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MODEL_DIR, TC_DIR, DEVICE, NUM_GENERATE
from pretrain_only import Generator


def main():
    print("=" * 60)
    print("  Step 3: Generate Test Cases")
    print("=" * 60)

    with open(os.path.join(DATA_DIR, "vocab.json"), 'r') as f:
        vocab = json.load(f)
        vocab = {int(k) if k.isdigit() else k: v for k, v in vocab.items()}
    max_len = int(open(os.path.join(DATA_DIR, "max_len.txt")).read().strip())
    pad_idx = vocab['<PAD>']
    vocab_size = max(vocab.values()) + 1

    with open(os.path.join(MODEL_DIR, "meta.json"), 'r') as f:
        meta = json.load(f)

    gen = Generator(
        vocab_size, meta["embed_dim"], meta["hidden_dim"],
        max_len, pad_idx
    )
    gen.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "generator_pretrained.pth"),
        map_location=DEVICE
    ))
    gen.to(DEVICE)

    inv_vocab = {v: k for k, v in vocab.items() if isinstance(k, int)}
    start_token = min(v for k, v in vocab.items() if isinstance(k, int) and v >= 2)

    print(f"Generating {NUM_GENERATE} samples (start_token={start_token})...")
    samples = gen.sample(NUM_GENERATE, start_token, DEVICE, max_len)

    all_cases = []
    valid_cases = []
    for seq in samples:
        byte_list = []
        for idx in seq:
            if idx == pad_idx:
                break
            if idx in inv_vocab:
                byte_list.append(inv_vocab[idx])
        if len(byte_list) >= 7:
            hex_msg = bytes(byte_list).hex()
            all_cases.append(hex_msg)
            fc = byte_list[7] if len(byte_list) > 7 else -1
            if fc in (1, 2, 3, 4, 5, 6, 15, 16):
                valid_cases.append(hex_msg)

    all_cases = list(dict.fromkeys(all_cases))
    valid_cases = list(dict.fromkeys(valid_cases))

    all_file = os.path.join(TC_DIR, "gan_all.txt")
    with open(all_file, 'w') as f:
        for c in all_cases:
            f.write(c + '\n')

    valid_file = os.path.join(TC_DIR, "gan_valid.txt")
    with open(valid_file, 'w') as f:
        for c in valid_cases:
            f.write(c + '\n')

    print(f"Generated: {len(all_cases)} total, {len(valid_cases)} valid (FC in 1-6,15,16)")
    print(f"Saved to {all_file}")
    print(f"Saved to {valid_file}")
    return len(all_cases), len(valid_cases)


if __name__ == "__main__":
    main()
