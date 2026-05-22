# preprocessing/clustering/no_clustering.py
from .base import process_and_save_cluster

def apply_no_clustering(messages_hex, max_len=None):
    """无聚类：所有消息作为一个整体"""
    process_and_save_cluster(messages_hex, "no_clustering", 0, max_len)