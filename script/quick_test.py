# scripts/quick_test.py
import os
import sys
import random
import socket
import time
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import TEST_CASES_DIR, TARGET_IP, TARGET_PORT, TEST_TIMEOUT
from utils.file_utils import load_text

def send_hex(hex_msg, ip, port, timeout):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((ip, port))
        sock.send(bytes.fromhex(hex_msg))
        resp = sock.recv(1024)
        sock.close()
        return True, resp.hex()
    except socket.timeout:
        return False, "timeout"
    except ConnectionRefusedError:
        return False, "connection refused"
    except Exception as e:
        return False, str(e)

def is_modbus_exception(response_hex):
    """检查响应是否为 Modbus 异常（功能码 >= 0x80）"""
    if not response_hex or len(response_hex) < 8:
        return False
    try:
        func_code = bytes.fromhex(response_hex)[7]
        return func_code >= 0x80
    except:
        return False

def quick_test(case_file, num_samples=100):
    cases = load_text(case_file)
    if not cases:
        print("No test cases found.")
        return
    if len(cases) > num_samples:
        samples = random.sample(cases, num_samples)
    else:
        samples = cases
    print(f"Testing {len(samples)} randomly selected cases...")
    results = []
    for idx, hex_msg in enumerate(samples):
        success, resp = send_hex(hex_msg, TARGET_IP, TARGET_PORT, TEST_TIMEOUT)
        if not success:
            results.append("TIMEOUT/REFUSED")
        elif is_modbus_exception(resp):
            results.append("MODBUS_EXCEPTION")
        else:
            results.append("ACCEPTED")
        if (idx+1) % 20 == 0:
            print(f"  Progress: {idx+1}/{len(samples)}")
    print("\n=== Quick Test Results ===")
    counter = Counter(results)
    total = len(samples)
    accepted = counter.get("ACCEPTED", 0)
    exception = counter.get("MODBUS_EXCEPTION", 0)
    refused = counter.get("TIMEOUT/REFUSED", 0)
    print(f"Accepted: {accepted} ({accepted/total*100:.1f}%)")
    print(f"Modbus Exception: {exception} ({exception/total*100:.1f}%)")
    print(f"Timeout/Refused: {refused} ({refused/total*100:.1f}%)")
    print(f"TIRR (Rejection Rate): {(exception+refused)/total*100:.1f}%")

if __name__ == "__main__":
    # 默认测试 GAN 生成的合并用例
    gan_file = os.path.join(TEST_CASES_DIR, "gan_cases.txt")
    if not os.path.exists(gan_file):
        print(f"File not found: {gan_file}")
        print("Please run model/generate_cases.py first.")
        sys.exit(1)
    quick_test(gan_file, num_samples=100)