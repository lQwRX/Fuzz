# fuzzer/fuzzer_main.py
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fuzzer.sender import ModbusSender
from fuzzer.logger import FuzzLogger
from utils.file_utils import load_text
from config import TARGET_IP, TARGET_PORT, TEST_TIMEOUT, LOG_DIR

def run_fuzzing(test_cases_file, log_file):
    cases = load_text(test_cases_file)
    if not cases:
        print("No test cases found.")
        return
    sender = ModbusSender(TARGET_IP, TARGET_PORT, TEST_TIMEOUT)
    logger = FuzzLogger(log_file)
    for idx, hex_msg in enumerate(cases):
        success, resp = sender.send_hex(hex_msg)
        status = "SUCCESS" if success else "TIMEOUT/REFUSED"
        logger.log(idx, hex_msg, status, resp)
        if (idx+1) % 500 == 0:
            print(f"Sent {idx+1}/{len(cases)}")
    logger.close()
    print(f"Log saved to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_cases", required=True)
    parser.add_argument("--log", default=os.path.join(LOG_DIR, "fuzz_gan.log"))
    args = parser.parse_args()
    run_fuzzing(args.test_cases, args.log)