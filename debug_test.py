import os, sys, json, torch, numpy as np, traceback
sys.path.append(os.path.dirname(os.path.abspath('.')))
sys.path.insert(0, '.')
from model.generator import Generator
from config import BASE_DIR, CLUSTER_DIR, OUTPUT_DIR, DEVICE

log_file = 'debug_gen_log.txt'
def log(msg):
    with open(log_file, 'a') as f:
        f.write(str(msg) + '\n')

log("START")

try:
    test_cases_dir = os.path.join(OUTPUT_DIR, "test_cases")
    os.makedirs(test_cases_dir, exist_ok=True)
    
    MODEL_CHECKPOINT_DIR = os.path.join(BASE_DIR, "model", "checkpoints_improved")
    log(f"checkpoint dir: {MODEL_CHECKPOINT_DIR}, exists: {os.path.exists(MODEL_CHECKPOINT_DIR)}")
    
    gen_file = os.path.join(MODEL_CHECKPOINT_DIR, "no_clustering", "class_0", "generator_step10.pth")
    log(f"gen_file: {gen_file}, exists: {os.path.exists(gen_file)}")
    
    data_path = os.path.join(CLUSTER_DIR, "no_clustering")
    log(f"data_path: {data_path}, exists: {os.path.exists(data_path)}")
    log(f"data.npy exists: {os.path.exists(os.path.join(data_path, 'data.npy'))}")
    
    vocab = json.load(open(os.path.join(data_path, "vocab.json")))
    vocab = {int(k) if k.isdigit() else k: v for k, v in vocab.items()}
    pad_idx = vocab['<PAD>']
    max_len = int(open(os.path.join(data_path, "max_len.txt")).read().strip())
    
    gen_ckpt = torch.load(gen_file, map_location=DEVICE)
    ckpt_vocab_size = gen_ckpt['embedding.weight'].shape[0]
    log(f"ckpt_vocab_size={ckpt_vocab_size}, max_len={max_len}, DEVICE={DEVICE}")
    
    gen = Generator(ckpt_vocab_size, 256, 32, max_len, pad_idx)
    gen.load_state_dict(gen_ckpt)
    gen.to(DEVICE)
    gen.eval()
    log("Generator loaded OK")
    
    log(f"Generating 10 samples...")
    samples = gen.sample(10, 2, DEVICE, max_len, temperature=1.2)
    log(f"Generated {len(samples)} samples")
    
    idx_to_byte = {}
    for k, v in vocab.items():
        if k == '<PAD>':
            continue
        idx_to_byte[v] = k
    
    hex_cases = []
    for seq in samples:
        byte_list = []
        for token in seq:
            if token == pad_idx:
                continue
            if token in idx_to_byte:
                byte_list.append(idx_to_byte[token])
        if len(byte_list) >= 7:
            hex_cases.append(bytes(byte_list).hex())
    
    log(f"Converted {len(hex_cases)} hex cases")
    
    out_file = os.path.join(test_cases_dir, "gan_cases.txt")
    with open(out_file, 'w') as f:
        for case in hex_cases:
            f.write(case + '\n')
    log(f"Wrote to {out_file}")
    log("SUCCESS")
except Exception as e:
    log(f"ERROR: {e}")
    log(traceback.format_exc())