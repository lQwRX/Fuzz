import os
import sys
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, "test")
OUTPUT_DIR = os.path.join(TEST_DIR, "output_test")

RAW_MSG_FILE = os.path.join(BASE_DIR, "data", "raw_messages.txt")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
TC_DIR = os.path.join(OUTPUT_DIR, "test_cases")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

MIN_MSG_LEN = 7
MAX_MSG_LEN = 64
BATCH_SIZE = 64
EMBED_DIM = 256
GEN_HIDDEN_DIM = 128
PRETRAIN_EPOCHS = 20
LR = 1e-3
NUM_GENERATE = 2000
NUM_BASELINE = 2000
TARGET_IP = "127.0.0.1"
TARGET_PORT = 502
TEST_TIMEOUT = 2

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for d in [DATA_DIR, MODEL_DIR, TC_DIR, LOG_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)
