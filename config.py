import os
import glob
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_MSG_FILE = os.path.join(DATA_DIR, "raw_messages.txt")
CLUSTER_DIR = os.path.join(DATA_DIR, "clusters")

PCAP_DIR = os.path.join(BASE_DIR, "pcap")
PCAP_PATHS = glob.glob(os.path.join(PCAP_DIR, "*.pcap"))

# 输出目录（用于评估和可视化）
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEST_CASES_DIR = os.path.join(OUTPUT_DIR, "test_cases")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# 是否自动生成模拟数据（当无pcap时）
AUTO_GENERATE_SYNTHETIC = True
SYNTHETIC_NUM_SAMPLES = 20000
SYNTHETIC_MAX_LEN = 100

# 数据增强：是否合并模拟数据到真实数据
ENABLE_SYNTHETIC_AUGMENT = True
AUGMENT_SYNTHETIC_COUNT = 15000   # 额外生成的模拟消息数量

# 数据质量改进：是否启用智能清洗（去重、FC03增广、分层平衡）
ENABLE_DATA_QUALITY_IMPROVEMENT = True
IMPROVED_MSG_FILE = os.path.join(DATA_DIR, "improved_raw_messages.txt")

# 预处理参数
MIN_MSG_LEN = 7
MAX_MSG_LEN = 64   # 建议设小一点，减少填充
REMOVE_DUPLICATES = True

# 聚类参数
SAME_LEN_THRESHOLD = 2000
ADVANCED_N_CLUSTERS = 8

# ========== 训练参数（补充） ==========
BATCH_SIZE = 64
EMBED_DIM = 256
GEN_HIDDEN_DIM = 32
DIS_NUM_FILTERS = 100
DIS_FILTER_SIZES = [2, 3, 4, 5]
PRETRAIN_GEN_EPOCHS = 5
PRETRAIN_DISC_EPOCHS = 5
ALT_STEPS = 10
ROLLOUT_NUM = 5
LR_G = 1e-4
LR_D = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 模糊测试评估参数 ==========
NUM_TARGET_CASES_PER_CLASS = 4200    # 每个聚类子集生成4200条测试用例
NUM_BASELINE_CASES = 50000           # 基线方法生成的测试用例数量
TARGET_IP = "127.0.0.1"
TARGET_PORT = 502
TEST_TIMEOUT = 2

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_CASES_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)