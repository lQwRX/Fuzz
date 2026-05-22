# preprocessing/clustering/same_length_clustering.py
from collections import defaultdict
from .base import process_and_save_cluster
from config import SAME_LEN_THRESHOLD

def apply_same_length_clustering(messages_hex, threshold=SAME_LEN_THRESHOLD, max_len=None):
    # 按原始消息长度（字节数）分组
    len_groups = defaultdict(list)
    for msg in messages_hex:
        l = len(msg) // 2
        len_groups[l].append(msg)

    # 合并小样本组到相邻更长的组
    sorted_lengths = sorted(len_groups.keys())
    merged = []
    current = []
    for l in sorted_lengths:
        if len(len_groups[l]) < threshold:
            current.extend(len_groups[l])
        else:
            if current:
                merged.append(current)
                current = []
            merged.append(len_groups[l])
    if current:
        merged.append(current)

    for idx, group in enumerate(merged):
        process_and_save_cluster(group, "same_length", idx, max_len)