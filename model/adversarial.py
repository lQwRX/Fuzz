import torch
from tqdm import tqdm
from .rollout import monte_carlo_search

def train_generator_step(gen, dis, train_loader, optimizer, rollout_num, device, pad_idx):
    gen.train()
    dis.eval()
    total_loss = 0
    loop = tqdm(train_loader, desc="Generator step", leave=False)
    for batch in loop:
        batch = batch.to(device)
        seq = batch
        rewards = monte_carlo_search(gen, dis, seq, rollout_num, device)
        logits, _ = gen(seq[:, :-1])
        log_probs = torch.log_softmax(logits, dim=-1)
        target = seq[:, 1:].unsqueeze(-1)
        log_prob_target = log_probs.gather(dim=-1, index=target).squeeze(-1)
        loss = - (log_prob_target * rewards[:, 1:]).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(train_loader)

def train_discriminator_step(dis, real_loader, fake_loader, optimizer, device):
    dis.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0
    loop = tqdm(zip(real_loader, fake_loader), total=len(real_loader), desc="Discriminator step", leave=False)
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
    return total_loss / len(real_loader)