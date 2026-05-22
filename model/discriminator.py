import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_filters=100, filter_sizes=[2,3,4,5], pad_idx=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)

    def forward(self, x):
        emb = self.embedding(x).unsqueeze(1)          # (batch, 1, seq_len, embed_dim)
        conv_out = []
        for conv in self.convs:
            c = torch.relu(conv(emb)).squeeze(3)      # (batch, num_filters, seq_len - fs + 1)
            p = torch.max_pool1d(c, c.size(2)).squeeze(2)  # (batch, num_filters)
            conv_out.append(p)
        feat = torch.cat(conv_out, dim=1)             # (batch, num_filters * len(filter_sizes))
        logit = self.fc(feat)
        return logit