import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR, MODEL_DIR, BATCH_SIZE, EMBED_DIM,
    GEN_HIDDEN_DIM, PRETRAIN_EPOCHS, LR, DEVICE
)


class MessageDataset(Dataset):
    def __init__(self, data):
        self.data = torch.LongTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Generator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_seq_len, pad_idx=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

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


def main():
    print("=" * 60)
    print("  Step 2: Pretrain Generator (MLE only)")
    print("=" * 60)

    data = np.load(os.path.join(DATA_DIR, "data.npy"))
    with open(os.path.join(DATA_DIR, "vocab.json"), 'r') as f:
        vocab = json.load(f)
        vocab = {int(k) if k.isdigit() else k: v for k, v in vocab.items()}
    max_len = int(open(os.path.join(DATA_DIR, "max_len.txt")).read().strip())

    pad_idx = vocab['<PAD>']
    vocab_size = max(vocab.values()) + 1
    print(f"Data: {data.shape}, vocab_size: {vocab_size}, max_len: {max_len}, pad_idx: {pad_idx}")
    print(f"Device: {DEVICE}, Hidden dim: {GEN_HIDDEN_DIM}, Epochs: {PRETRAIN_EPOCHS}")

    dataset = MessageDataset(data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    gen = Generator(vocab_size, EMBED_DIM, GEN_HIDDEN_DIM, max_len, pad_idx).to(DEVICE)
    optimizer = torch.optim.Adam(gen.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    print(f"\nGenerator parameters: {sum(p.numel() for p in gen.parameters()):,}")
    print(f"Starting pretraining for {PRETRAIN_EPOCHS} epochs...\n")

    for epoch in range(PRETRAIN_EPOCHS):
        gen.train()
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{PRETRAIN_EPOCHS}", leave=False)
        for batch in loop:
            batch = batch.to(DEVICE)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits, _ = gen(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{PRETRAIN_EPOCHS}  avg_loss={avg:.4f}")

    model_path = os.path.join(MODEL_DIR, "generator_pretrained.pth")
    torch.save(gen.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    meta = {
        "vocab_size": vocab_size,
        "embed_dim": EMBED_DIM,
        "hidden_dim": GEN_HIDDEN_DIM,
        "max_len": max_len,
        "pad_idx": pad_idx,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "final_loss": avg
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    return model_path


if __name__ == "__main__":
    main()
