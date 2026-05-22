# feedback/mutate_and_retrain.py
import os
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------- 配置 ----------
BASE_DIR = r"D:\PythonWorkSpace\Fuzz"
CLUSTER_DIR = os.path.join(BASE_DIR, "data", "clusters")
MODEL_DIR = os.path.join(BASE_DIR, "model", "checkpoints_improved", "no_clustering", "class_0")
SEED_FILE = os.path.join(BASE_DIR, "valid_output", "anomaly_seeds.txt")
OUTPUT_MODEL = os.path.join(MODEL_DIR, "retrain", "generator_retrained.pth")
BATCH_SIZE = 64
EMBED_DIM = 256
GEN_HIDDEN_DIM = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Dataset ----------
class MessageDataset(Dataset):
    def __init__(self, data):
        self.data = torch.LongTensor(data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# ---------- Generator ----------
class Generator(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_seq_len, pad_idx=1):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

    def sample(self, batch_size, start_token, device, max_len=None):
        if max_len is None:
            max_len = self.max_seq_len
        self.eval()
        samples = []
        for _ in range(batch_size):
            seq = [start_token]
            input_t = torch.tensor([[start_token]], device=device)
            h = None
            for _ in range(max_len - 1):
                logits, h = self.forward(input_t, h)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                seq.append(next_token)
                input_t = torch.tensor([[next_token]], device=device)
            samples.append(seq)
        return samples

# ---------- 辅助函数 ----------
def mutate_hex(hex_msg, num_flips=1):
    data = bytearray.fromhex(hex_msg)
    if not data:
        return hex_msg
    for _ in range(num_flips):
        pos = random.randint(0, len(data)-1)
        data[pos] ^= random.randint(1, 255)
    return data.hex()

def hex_to_sequence(hex_msg, vocab, pad_idx, max_len):
    byte_data = bytes.fromhex(hex_msg)
    seq = [vocab.get(b, pad_idx) for b in byte_data]
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq += [pad_idx] * (max_len - len(seq))
    return seq

# ---------- 主流程 ----------
def main():
    if not os.path.exists(SEED_FILE):
        print(f"Seed file not found: {SEED_FILE}")
        print("Please run extract_anomalies_fixed.py first.")
        return

    # 加载原始训练数据
    data_path = os.path.join(CLUSTER_DIR, "no_clustering")
    original_data = np.load(os.path.join(data_path, "data.npy"))
    with open(os.path.join(data_path, "vocab.json"), 'r') as f:
        vocab = json.load(f)
    vocab = {int(k) if k.isdigit() else k: v for k, v in vocab.items()}
    pad_idx = vocab['<PAD>']
    max_len = int(open(os.path.join(data_path, "max_len.txt")).read().strip())
    max_idx = original_data.max()
    vocab_size = max_idx + 1
    print(f"Original data shape: {original_data.shape}, vocab_size: {vocab_size}, max_len: {max_len}")

    # 加载高价值异常种子
    with open(SEED_FILE, 'r') as f:
        seeds = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(seeds)} high-value anomaly seeds")

    if not seeds:
        return

    # 突变扩展（每条种子生成 5 个变体）
    expanded = []
    for seed in seeds[:500]:  # 限制数量，避免数据集过大
        expanded.append(seed)
        for _ in range(5):
            mutated = mutate_hex(seed, random.randint(1,2))
            if len(mutated)//2 > max_len:
                mutated = mutated[:max_len*2]
            expanded.append(mutated)
    expanded = list(dict.fromkeys(expanded))
    print(f"Expanded to {len(expanded)} samples")

    # 转换为序列
    new_seqs = [hex_to_sequence(h, vocab, pad_idx, max_len) for h in expanded]
    new_data = np.array(new_seqs, dtype=np.int64)
    combined_data = np.vstack([original_data, new_data])
    print(f"Combined data shape: {combined_data.shape}")

    # 加载原始生成器
    model_file = os.path.join(MODEL_DIR, "generator_step10.pth")
    if not os.path.exists(model_file):
        print(f"Model not found: {model_file}")
        return
    gen = Generator(vocab_size, EMBED_DIM, GEN_HIDDEN_DIM, max_len, pad_idx)
    gen.load_state_dict(torch.load(model_file, map_location=DEVICE))
    gen.to(DEVICE)

    # 微调 2~3 个 epoch
    dataset = MessageDataset(combined_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    gen.train()
    for epoch in range(2):
        total_loss = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits, _ = gen(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(loader):.4f}")

    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    torch.save(gen.state_dict(), OUTPUT_MODEL)
    print(f"Retrained model saved to {OUTPUT_MODEL}")

if __name__ == "__main__":
    main()