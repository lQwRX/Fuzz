#生成反馈调节板测试用例
# model/generate_from_retrained.py
import os
import sys
import json
import torch
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.generator import Generator
from config import BASE_DIR, CLUSTER_DIR, NUM_TARGET_CASES_PER_CLASS, DEVICE

BATCH_CHUNK = 500


def is_valid_modbus_request(hex_str: str) -> bool:
    if len(hex_str) < 14:
        return False
    try:
        func_code = int(hex_str[14:16], 16)
        return func_code in (1,2,3,4,5,6,15,16)
    except:
        return False


def load_cluster_info(data_path):
    with open(os.path.join(data_path, "vocab.json"), 'r') as f:
        vocab = json.load(f)
    vocab = {int(k) if k.isdigit() else k: v for k, v in vocab.items()}
    pad_idx = vocab['<PAD>']
    max_len = int(open(os.path.join(data_path, "max_len.txt")).read().strip())
    data = np.load(os.path.join(data_path, "data.npy"))
    max_idx = data.max()
    vocab_size = max_idx + 1
    return vocab, pad_idx, max_len, vocab_size


def generate_valid_cases_from_model(
    model_path, data_path, num_samples,
    start_token=2, output_path=None, resume=False
):
    vocab, pad_idx, max_len, vocab_size = load_cluster_info(data_path)
    idx_to_byte = {v: k for k, v in vocab.items() if k != '<PAD>'}

    gen = Generator(vocab_size, 256, 32, max_len, pad_idx)
    gen.load_state_dict(torch.load(model_path, map_location=DEVICE))
    gen.to(DEVICE)
    gen.eval()

    existing = set()
    if resume and output_path and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(line)
        print(f"Resuming: {len(existing)} cases already exist in {output_path}")

    all_cases = list(existing)
    needed = num_samples - len(all_cases)
    if needed <= 0:
        print(f"Already have {len(all_cases)} >= {num_samples}, nothing to generate.")
        return all_cases

    print(f"Need {needed} more cases, generating in chunks of {BATCH_CHUNK}...")
    generated = 0
    while generated < needed:
        cur_batch = min(BATCH_CHUNK, needed - generated)
        samples = gen.sample(cur_batch, start_token, DEVICE, max_len)

        for seq in samples:
            if len(seq) > 3:
                seq[2] = 2
                seq[3] = 2

        batch_hex = []
        for seq in samples:
            bytes_list = [idx_to_byte[idx] for idx in seq if idx != pad_idx and idx in idx_to_byte]
            if len(bytes_list) < 7:
                continue
            hex_msg = bytes(bytes_list).hex()
            if is_valid_modbus_request(hex_msg):
                batch_hex.append(hex_msg)

        # Deduplicate within batch and against existing
        new_in_batch = []
        for h in batch_hex:
            if h not in all_cases:
                new_in_batch.append(h)
                all_cases.append(h)

        generated += len(batch_hex)
        print(f"  chunk {generated}/{needed}: +{len(new_in_batch)} new (batch produced {len(batch_hex)} valid)")

        # Save incrementally after each chunk
        if output_path and new_in_batch:
            with open(output_path, 'a') as f:
                for h in new_in_batch:
                    f.write(h + '\n')
            print(f"    saved to {output_path}")

    print(f"Done. Total unique cases: {len(all_cases)}")
    return all_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to retrained generator .pth")
    parser.add_argument("--data_path", default=None, help="Path to cluster data")
    parser.add_argument("--num_samples", type=int, default=NUM_TARGET_CASES_PER_CLASS)
    parser.add_argument("--output", default=None, help="Output file path (default: valid_output/test_cases/gan_cases_retrained.txt)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = os.path.join(CLUSTER_DIR, "no_clustering")

    if args.output is None:
        out_dir = os.path.join(BASE_DIR, "valid_output", "test_cases")
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, "gan_cases_retrained.txt")

    generate_valid_cases_from_model(
        args.model_path, args.data_path, args.num_samples,
        output_path=args.output, resume=args.resume
    )


if __name__ == "__main__":
    main()