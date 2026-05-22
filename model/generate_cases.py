import os
import sys
import json
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.generator import Generator
from config import (BASE_DIR, CLUSTER_DIR, OUTPUT_DIR, DEVICE,
                    NUM_TARGET_CASES_PER_CLASS)


def log(msg):
    with open(os.path.join(OUTPUT_DIR, "test_cases", "gen_log.txt"), 'a') as f:
        f.write(str(msg) + '\n')


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


def generate_for_cluster(model_path, data_path, num_samples,
                          start_token=2, temperature=1.2):
    vocab, pad_idx, max_len, vocab_size = load_cluster_info(data_path)

    idx_to_byte = {}
    for k, v in vocab.items():
        if k == '<PAD>':
            continue
        idx_to_byte[v] = k

    gen_ckpt = torch.load(model_path, map_location=DEVICE)
    ckpt_vocab_size = gen_ckpt['embedding.weight'].shape[0]
    if ckpt_vocab_size != vocab_size:
        log(f"  vocab_size mismatch (ckpt={ckpt_vocab_size}, data={vocab_size}), using ckpt value")

    gen = Generator(ckpt_vocab_size, 256, 32, max_len, pad_idx)
    gen.load_state_dict(gen_ckpt)
    gen.to(DEVICE)
    gen.eval()

    log(f"  Generating {num_samples} samples (batch=32)...")
    samples = []
    remaining = num_samples
    while remaining > 0:
        cur_batch = min(500, remaining)
        batch = gen.sample(cur_batch, start_token, DEVICE, max_len,
                            temperature=temperature)
        samples.extend(batch)
        remaining -= cur_batch
    log(f"  Generated {len(samples)} raw samples")

    # Force Protocol ID (bytes 2-3) to 0x0000 for Modbus TCP compliance
    # byte 0x00 maps to index 2 in the vocab
    for seq in samples:
        if len(seq) > 3:
            seq[2] = 2
            seq[3] = 2

    hex_cases = []
    for seq in samples:
        byte_list = []
        for token in seq:
            if token == pad_idx:
                continue
            if token in idx_to_byte:
                byte_list.append(idx_to_byte[token])
            else:
                continue
        if len(byte_list) < 7:
            continue
        hex_cases.append(bytes(byte_list).hex())

    log(f"  Converted to {len(hex_cases)} hex cases (before dedup)")
    return list(dict.fromkeys(hex_cases))


def main():
    test_cases_dir = os.path.join(OUTPUT_DIR, "test_cases")
    os.makedirs(test_cases_dir, exist_ok=True)
    log("=== Generation started ===")

    all_cases = []
    cluster_cases = {
        "no_clustering": [],
        "same_length": [],
        "advanced": []
    }

    MODEL_CHECKPOINT_DIR = os.path.join(BASE_DIR, "model", "checkpoints_improved")
    log(f"MODEL_CHECKPOINT_DIR={MODEL_CHECKPOINT_DIR} exists={os.path.exists(MODEL_CHECKPOINT_DIR)}")
    if not os.path.exists(MODEL_CHECKPOINT_DIR):
        log("Model directory not found!")
        return

    for cluster_name in ["no_clustering", "same_length", "advanced"]:
        cluster_model_dir = os.path.join(MODEL_CHECKPOINT_DIR, cluster_name)
        log(f"Processing {cluster_name}, dir exists={os.path.exists(cluster_model_dir)}")
        if not os.path.exists(cluster_model_dir):
            continue

        if cluster_name == "no_clustering":
            gen_file = os.path.join(cluster_model_dir, "class_0", "generator_step10.pth")
            if not os.path.exists(gen_file):
                log(f"Model not found: {gen_file}")
                continue
            data_path = os.path.join(CLUSTER_DIR, cluster_name)
            log(f"Generating from {cluster_name} ...")
            try:
                cases = generate_for_cluster(gen_file, data_path, NUM_TARGET_CASES_PER_CLASS)
                log(f"  Got {len(cases)} cases")
                cluster_cases[cluster_name].extend(cases)
                all_cases.extend(cases)
            except Exception as e:
                log(f"  ERROR: {e}")
                import traceback
                log(traceback.format_exc())
        else:
            subdirs = [os.path.join(cluster_model_dir, d)
                       for d in os.listdir(cluster_model_dir) if d.startswith("class_")]
            log(f"  subdirs={[os.path.basename(d) for d in subdirs]}")
            for subdir in subdirs:
                gen_file = os.path.join(subdir, "generator_step10.pth")
                if not os.path.exists(gen_file):
                    continue
                class_idx = os.path.basename(subdir).split("_")[-1]
                data_path = os.path.join(CLUSTER_DIR, cluster_name, f"class_{class_idx}")
                log(f"Generating from {cluster_name}/class_{class_idx} ...")
                try:
                    cases = generate_for_cluster(gen_file, data_path, NUM_TARGET_CASES_PER_CLASS)
                    log(f"  Got {len(cases)} cases")
                    cluster_cases[cluster_name].extend(cases)
                    all_cases.extend(cases)
                except Exception as e:
                    log(f"  ERROR: {e}")
                    import traceback
                    log(traceback.format_exc())

    all_cases = list(dict.fromkeys(all_cases))
    for k in cluster_cases:
        cluster_cases[k] = list(dict.fromkeys(cluster_cases[k]))

    log(f"Total (after dedup): {len(all_cases)}")
    merged_file = os.path.join(test_cases_dir, "gan_cases.txt")
    with open(merged_file, 'w') as f:
        for case in all_cases:
            f.write(case + '\n')
    log(f"Saved to {merged_file}")

    for cluster_name, cases in cluster_cases.items():
        if cases:
            out_file = os.path.join(test_cases_dir, f"gan_{cluster_name}.txt")
            with open(out_file, 'w') as f:
                for case in cases:
                    f.write(case + '\n')
            log(f"{cluster_name}: {len(cases)} cases saved to {out_file}")
        else:
            log(f"{cluster_name}: no test cases generated.")

    log("=== Generation finished ===")


if __name__ == "__main__":
    main()