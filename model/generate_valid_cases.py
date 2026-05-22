#生成加强版测试用例
# model/generate_valid_cases.py
import os
import sys
import json
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.generator import Generator
from config import BASE_DIR, CLUSTER_DIR, NUM_TARGET_CASES_PER_CLASS, DEVICE

def is_valid_modbus_request(hex_str: str) -> bool:
    if len(hex_str) < 14:
        return False
    try:
        func_code = int(hex_str[14:16], 16)
        return func_code in (1, 2, 3, 4, 5, 6, 15, 16)
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

def generate_valid_cases_for_cluster(model_path, data_path, num_samples, start_token=2):
    vocab, pad_idx, max_len, vocab_size = load_cluster_info(data_path)
    idx_to_byte = {v: k for k, v in vocab.items() if k != '<PAD>'}
    gen = Generator(vocab_size, 256, 32, max_len, pad_idx)
    gen.load_state_dict(torch.load(model_path, map_location=DEVICE))
    gen.to(DEVICE)
    gen.eval()

    samples = gen.sample(num_samples, start_token, DEVICE, max_len)

    for seq in samples:
        if len(seq) > 3:
            seq[2] = 2
            seq[3] = 2

    valid_cases = []
    for seq in samples:
        byte_list = []
        for idx in seq:
            if idx == pad_idx or idx not in idx_to_byte:
                continue
            byte_list.append(idx_to_byte[idx])
        if len(byte_list) < 7:
            continue
        hex_msg = bytes(byte_list).hex()
        if is_valid_modbus_request(hex_msg):
            valid_cases.append(hex_msg)
    return list(dict.fromkeys(valid_cases))

def main():
    print("Generating valid test cases (filtered by Modbus function codes)...")
    valid_root = os.path.join(BASE_DIR, "valid_output")
    test_cases_dir = os.path.join(valid_root, "test_cases")
    os.makedirs(test_cases_dir, exist_ok=True)

    all_valid_cases = []
    cluster_cases = {"no_clustering": [], "same_length": [], "advanced": []}
    MODEL_CHECKPOINT_DIR = os.path.join(BASE_DIR, "model", "checkpoints_improved")
    if not os.path.exists(MODEL_CHECKPOINT_DIR):
        print(f"Error: {MODEL_CHECKPOINT_DIR} not found.")
        return

    for cluster_name in ["no_clustering", "same_length", "advanced"]:
        cluster_model_dir = os.path.join(MODEL_CHECKPOINT_DIR, cluster_name)
        if not os.path.exists(cluster_model_dir):
            continue
        if cluster_name == "no_clustering":
            model_file = os.path.join(cluster_model_dir, "class_0", "generator_step10.pth")
            if not os.path.exists(model_file):
                print(f"Model not found: {model_file}")
                continue
            data_path = os.path.join(CLUSTER_DIR, cluster_name)
            print(f"Generating from {cluster_name} ...")
            cases = generate_valid_cases_for_cluster(model_file, data_path, NUM_TARGET_CASES_PER_CLASS)
            cluster_cases[cluster_name].extend(cases)
            all_valid_cases.extend(cases)
        else:
            subdirs = [os.path.join(cluster_model_dir, d) for d in os.listdir(cluster_model_dir) if d.startswith("class_")]
            for subdir in subdirs:
                model_file = os.path.join(subdir, "generator_step10.pth")
                if not os.path.exists(model_file):
                    continue
                class_idx = os.path.basename(subdir).split("_")[-1]
                data_path = os.path.join(CLUSTER_DIR, cluster_name, f"class_{class_idx}")
                print(f"Generating from {cluster_name}/class_{class_idx} ...")
                cases = generate_valid_cases_for_cluster(model_file, data_path, NUM_TARGET_CASES_PER_CLASS)
                cluster_cases[cluster_name].extend(cases)
                all_valid_cases.extend(cases)

    all_valid_cases = list(dict.fromkeys(all_valid_cases))
    for k in cluster_cases:
        cluster_cases[k] = list(dict.fromkeys(cluster_cases[k]))

    merged_file = os.path.join(test_cases_dir, "gan_cases.txt")
    with open(merged_file, 'w') as f:
        for case in all_valid_cases:
            f.write(case + '\n')
    print(f"Total valid test cases: {len(all_valid_cases)} saved to {merged_file}")

    for cluster_name, cases in cluster_cases.items():
        if cases:
            out_file = os.path.join(test_cases_dir, f"gan_{cluster_name}.txt")
            with open(out_file, 'w') as f:
                for case in cases:
                    f.write(case + '\n')
            print(f"{cluster_name}: {len(cases)} valid test cases saved to {out_file}")
        else:
            print(f"{cluster_name}: no valid test cases generated.")

if __name__ == "__main__":
    main()