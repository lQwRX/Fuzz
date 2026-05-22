# preprocessing/clustering/advanced_clustering.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from .base import process_and_save_cluster
from config import ADVANCED_N_CLUSTERS, MAX_MSG_LEN
from utils.conversion_utils import hex_to_bytes

def apply_advanced_clustering(messages_hex, n_clusters=ADVANCED_N_CLUSTERS, max_len=MAX_MSG_LEN):
    # 将所有消息填充到相同长度（用于向量化）
    vectors = []
    for msg in messages_hex:
        b = hex_to_bytes(msg)
        if len(b) > max_len:
            b = b[:max_len]
        # 转为 uint8 数组，不足补0
        arr = np.frombuffer(b, dtype=np.uint8) if len(b)>0 else np.array([], dtype=np.uint8)
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)), constant_values=0)
        vectors.append(arr)
    vectors = np.array(vectors, dtype=np.float32)
    # 归一化
    vectors = normalize(vectors, norm='l2')
    # k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)

    # 按标签分组
    groups = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        groups[label].append(messages_hex[idx])

    # 保存每个聚类子集
    for idx, group in enumerate(groups):
        if len(group) > 0:
            process_and_save_cluster(group, "advanced", idx, max_len)