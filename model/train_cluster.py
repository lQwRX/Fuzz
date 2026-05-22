import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .generator import Generator
from .discriminator import Discriminator
from .dataset import MessageDataset
from .pretrain import pretrain_generator, pretrain_discriminator
from .adversarial import train_generator_step, train_discriminator_step

# 配置参数（可根据需要修改）
BATCH_SIZE = 64
EMBED_DIM = 256
GEN_HIDDEN_DIM = 32
DIS_NUM_FILTERS = 100
DIS_FILTER_SIZES = [2,3,4,5]
PRETRAIN_GEN_EPOCHS = 5
PRETRAIN_DISC_EPOCHS = 5
ALT_STEPS = 10
ROLLOUT_NUM = 5
LR_G = 1e-4
LR_D = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_single_cluster(data_path, output_model_dir, cluster_name, class_idx):
    """
    data_path: 包含 data.npy, vocab.json, max_len.txt 的目录
    output_model_dir: 保存模型的根目录 (例如 'model/checkpoints_improved')
    cluster_name, class_idx: 用于命名
    """
    print(f"\n========== Training {cluster_name}/class_{class_idx} ==========")
    with tqdm(total=1, desc="Loading data", leave=False) as pbar:
        data = np.load(os.path.join(data_path, "data.npy"))
        with open(os.path.join(data_path, "vocab.json"), 'r') as f:
            vocab = json.load(f)
            # 将键转换为整数（如果键是字符串数字）
            vocab = {int(k) if k.isdigit() else k: v for k, v in vocab.items()}
        max_len = int(open(os.path.join(data_path, "max_len.txt")).read().strip())
        pbar.update(1)

    max_idx = data.max()
    pad_idx = vocab['<PAD>']
    vocab_size = max_idx + 1          # 关键修正
    print(f"Data shape {data.shape}, vocab_size {vocab_size}, max_len {max_len}, pad_idx {pad_idx}")

    dataset = MessageDataset(data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    gen = Generator(vocab_size, EMBED_DIM, GEN_HIDDEN_DIM, max_len, pad_idx)
    dis = Discriminator(vocab_size, EMBED_DIM, max_len, DIS_NUM_FILTERS, DIS_FILTER_SIZES, pad_idx)

    # 预训练生成器
    pretrain_generator(gen, loader, PRETRAIN_GEN_EPOCHS, lr=1e-3, device=DEVICE, pad_idx=pad_idx)

    # 生成假样本用于预训练判别器
    tqdm.write("Generating fake samples for discriminator pre-training...")
    start_token = 2  # 根据 vocab 中最小有效索引确定，通常为2
    num_samples = len(dataset)
    fake_samples = []
    batch_size_gen = 1000
    with tqdm(total=num_samples, desc="Generating", leave=False) as pbar:
        for start in range(0, num_samples, batch_size_gen):
            cur_batch = min(batch_size_gen, num_samples - start)
            batch_samples = gen.sample(cur_batch, start_token, DEVICE, max_len)
            fake_samples.extend(batch_samples)
            pbar.update(cur_batch)
    padded_fake = []
    for seq in fake_samples:
        if len(seq) > max_len:
            seq = seq[:max_len]
        padded_fake.append(seq + [pad_idx] * (max_len - len(seq)))
    fake_dataset = MessageDataset(padded_fake)
    fake_loader = DataLoader(fake_dataset, batch_size=BATCH_SIZE, shuffle=True)
    pretrain_discriminator(dis, loader, fake_loader, PRETRAIN_DISC_EPOCHS, lr=1e-3, device=DEVICE)

    # 对抗交替训练
    opt_g = torch.optim.Adam(gen.parameters(), lr=LR_G)
    opt_d = torch.optim.Adam(dis.parameters(), lr=LR_D)
    for step in range(1, ALT_STEPS+1):
        tqdm.write(f"\n--- Alternating step {step} ---")
        gen_loss = train_generator_step(gen, dis, loader, opt_g, ROLLOUT_NUM, DEVICE, pad_idx)
        tqdm.write(f"Generator loss: {gen_loss:.4f}")

        # 重新生成假样本用于判别器
        tqdm.write("Generating fresh fake samples...")
        fake_samples = []
        with tqdm(total=num_samples, desc="Sampling", leave=False) as pbar:
            for start in range(0, num_samples, batch_size_gen):
                cur_batch = min(batch_size_gen, num_samples - start)
                batch_samples = gen.sample(cur_batch, start_token, DEVICE, max_len)
                fake_samples.extend(batch_samples)
                pbar.update(cur_batch)
        padded_fake = []
        for seq in fake_samples:
            if len(seq) > max_len:
                seq = seq[:max_len]
            padded_fake.append(seq + [pad_idx] * (max_len - len(seq)))
        fake_dataset = MessageDataset(padded_fake)
        fake_loader = DataLoader(fake_dataset, batch_size=BATCH_SIZE, shuffle=True)

        for disc_epoch in range(5):
            disc_loss = train_discriminator_step(dis, loader, fake_loader, opt_d, DEVICE)
            tqdm.write(f"Discriminator loss (epoch {disc_epoch+1}): {disc_loss:.4f}")

        if step % 5 == 0 or step == ALT_STEPS:
            save_dir = os.path.join(output_model_dir, cluster_name, f"class_{class_idx}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(gen.state_dict(), os.path.join(save_dir, f"generator_step{step}.pth"))
            torch.save(dis.state_dict(), os.path.join(save_dir, f"discriminator_step{step}.pth"))
            tqdm.write(f"Saved model at step {step}")
    tqdm.write(f"Finished {cluster_name}/class_{class_idx}")