import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def pretrain_generator(gen, train_loader, epochs, lr, device, pad_idx):
    optimizer = torch.optim.Adam(gen.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    gen.to(device)
    gen.train()
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Pretrain Gen E{epoch+1}", leave=False)
        for batch in loop:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits, _ = gen(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_loader)
        tqdm.write(f"Pretrain Gen E{epoch+1} avg loss: {avg_loss:.4f}")

def pretrain_discriminator(dis, real_loader, fake_loader, epochs, lr, device):
    optimizer = torch.optim.Adam(dis.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    dis.to(device)
    dis.train()
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(zip(real_loader, fake_loader), total=len(real_loader), desc=f"Pretrain Disc E{epoch+1}", leave=False)
        for real_batch, fake_batch in loop:
            real_batch = real_batch.to(device)
            fake_batch = fake_batch.to(device)
            real_labels = torch.ones(real_batch.size(0), 1, device=device)
            fake_labels = torch.zeros(fake_batch.size(0), 1, device=device)
            logits_real = dis(real_batch)
            logits_fake = dis(fake_batch)
            loss = criterion(logits_real, real_labels) + criterion(logits_fake, fake_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(real_loader)
        tqdm.write(f"Pretrain Disc E{epoch+1} avg loss: {avg_loss:.4f}")
