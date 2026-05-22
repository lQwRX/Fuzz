# scripts/run_preprocessing.py
import sys

sys.path.append('..')
from capture.message_collector import MessageCollector
from preprocessing import Cleaner, apply_no_clustering, apply_same_length_clustering, apply_advanced_clustering
from utils.file_utils import load_text, save_text
from config import (
    RAW_MSG_FILE, ENABLE_SYNTHETIC_AUGMENT, AUGMENT_SYNTHETIC_COUNT,
    SYNTHETIC_MAX_LEN, CLUSTER_DIR, ENABLE_DATA_QUALITY_IMPROVEMENT, IMPROVED_MSG_FILE
)
import os
import shutil

if __name__ == "__main__":
    # 1. 如果启用数据质量改进且有改进文件，直接使用
    if ENABLE_DATA_QUALITY_IMPROVEMENT and os.path.exists(IMPROVED_MSG_FILE):
        print(f"Loading improved messages from {IMPROVED_MSG_FILE}")
        cleaned = load_text(IMPROVED_MSG_FILE)
        print(f"Loaded {len(cleaned)} quality-improved messages.")
    else:
        # 2. 收集真实消息或加载已有 raw_messages.txt
        if os.path.exists(RAW_MSG_FILE):
            print(f"Loading existing raw messages from {RAW_MSG_FILE}")
            real_msgs = load_text(RAW_MSG_FILE)
            print(f"Loaded {len(real_msgs)} real messages.")
        else:
            real_msgs = MessageCollector.collect()

        # 3. 合并模拟数据（如果启用）
        if ENABLE_SYNTHETIC_AUGMENT:
            print(f"Generating {AUGMENT_SYNTHETIC_COUNT} synthetic messages for augmentation...")
            synthetic_msgs = MessageCollector.generate_synthetic_modbus(AUGMENT_SYNTHETIC_COUNT, SYNTHETIC_MAX_LEN)
            all_msgs = real_msgs + synthetic_msgs
            print(f"Total messages after augmentation: {len(all_msgs)} (real: {len(real_msgs)}, synthetic: {len(synthetic_msgs)})")
            merged_file = os.path.join(os.path.dirname(RAW_MSG_FILE), "merged_raw_messages.txt")
            save_text(all_msgs, merged_file)
            print(f"Merged messages saved to {merged_file}")
        else:
            all_msgs = real_msgs

        # 4. 清洗
        cleaned = Cleaner.clean(all_msgs)
        print(f"Cleaned messages: {len(cleaned)}")

    # 5. 删除旧的 clusters 目录
    if os.path.exists(CLUSTER_DIR):
        print(f"Removing old cluster directory: {CLUSTER_DIR}")
        shutil.rmtree(CLUSTER_DIR)

    # 6. 三种聚类
    print("Applying NoClustering...")
    apply_no_clustering(cleaned)
    print("Applying SameLengthClustering...")
    apply_same_length_clustering(cleaned)
    print("Applying AdvancedClustering...")
    apply_advanced_clustering(cleaned)
    print("Preprocessing done. Check data/clusters/")