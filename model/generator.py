import torch
import torch.nn as nn

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

    def sample(self, num_samples, start_token, device, max_len=None, temperature=1.0):
        if max_len is None:
            max_len = self.max_seq_len
        self.eval()
        batch_size = min(32, num_samples)
        all_samples = []
        generated = 0
        while generated < num_samples:
            cur_batch = min(batch_size, num_samples - generated)
            batch_input = torch.full((cur_batch, 1), start_token, dtype=torch.long, device=device)
            h = None
            for _ in range(max_len - 1):
                logits, h = self.forward(batch_input[:, -1:], h)
                scaled_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1)
                batch_input = torch.cat([batch_input, next_tokens], dim=1)
            all_samples.extend(batch_input.tolist())
            generated += cur_batch
        return all_samples