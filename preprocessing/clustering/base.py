# preprocessing/clustering/base.py
import os
import numpy as np
from preprocessing.vocab_builder import VocabBuilder
from preprocessing.numericalizer import Numericalizer
from preprocessing.padder import Padder
from utils.file_utils import ensure_dir, save_json
from config import CLUSTER_DIR

def save_cluster(cluster_name, cluster_idx, data, vocab, max_len):
    """保存单个聚类子集到磁盘"""
    if cluster_name == "no_clustering":
        subdir = os.path.join(CLUSTER_DIR, cluster_name)
    else:
        subdir = os.path.join(CLUSTER_DIR, cluster_name, f"class_{cluster_idx}")
    ensure_dir(subdir)
    np.save(os.path.join(subdir, "data.npy"), data)
    save_json(vocab, os.path.join(subdir, "vocab.json"))
    with open(os.path.join(subdir, "max_len.txt"), 'w') as f:
        f.write(str(max_len))
    print(f"Saved cluster {cluster_name}/{cluster_idx} with shape {data.shape}")

def process_and_save_cluster(messages_hex, cluster_name, cluster_idx, max_len=None):
    """对一组消息进行 vocab 构建、数值化、填充，然后保存"""
    if not messages_hex:
        return
    vocab = VocabBuilder.build_vocab(messages_hex)
    pad_idx = VocabBuilder.get_pad_idx()
    seqs = Numericalizer.messages_to_sequences(messages_hex, vocab)
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    max_len = min(max_len, 260)   # 限制
    seqs = Padder.truncate_sequences(seqs, max_len)
    data = Padder.pad_sequences(seqs, max_len, pad_val=pad_idx)
    save_cluster(cluster_name, cluster_idx, data, vocab, max_len)