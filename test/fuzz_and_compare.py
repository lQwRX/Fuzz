import os
import sys
import json
import random
import socket
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TC_DIR, LOG_DIR, REPORT_DIR, RAW_MSG_FILE,
    NUM_BASELINE, TARGET_IP, TARGET_PORT, TEST_TIMEOUT
)


def load_text(filepath):
    with open(filepath, 'r') as f:
        return [l.strip() for l in f if l.strip()]


def generate_random_cases(num, min_len=7, max_len=64):
    cases = []
    for _ in range(num):
        length = random.randint(min_len, max_len)
        raw = bytes([random.randint(0, 255) for _ in range(length)])
        cases.append(raw.hex())
    return cases


def generate_mutation_cases(seed_file, num, flips=3):
    seeds = load_text(seed_file)
    if not seeds:
        seeds = ["00010000000601030000000A"] * 100
    cases = []
    while len(cases) < num:
        seed = random.choice(seeds)
        data = bytearray.fromhex(seed)
        for _ in range(random.randint(1, flips)):
            pos = random.randint(0, len(data) - 1)
            data[pos] ^= random.randint(1, 255)
        cases.append(data.hex())
        if len(cases) >= num:
            break
    return cases[:num]


def send_hex(hex_msg, ip, port, timeout):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((ip, port))
        sock.send(bytes.fromhex(hex_msg))
        response = sock.recv(1024)
        sock.close()
        return True, response.hex()
    except Exception as e:
        return False, str(e)


def is_modbus_exception(response_hex):
    if not response_hex or len(response_hex) < 8:
        return False
    try:
        resp_bytes = bytes.fromhex(response_hex)
        if len(resp_bytes) < 8:
            return False
        return resp_bytes[7] >= 0x80
    except:
        return False


def run_fuzzing(test_cases, ip, port, timeout, log_file):
    with open(log_file, 'w') as f:
        f.write("timestamp\tindex\thex_msg\tstatus\tresponse\n")
        for idx, hex_msg in enumerate(test_cases):
            success, resp = send_hex(hex_msg, ip, port, timeout)
            status = "SUCCESS" if success else "TIMEOUT/REFUSED"
            f.write(f"{time.time()}\t{idx}\t{hex_msg}\t{status}\t{resp}\n")
            f.flush()
            if (idx + 1) % 500 == 0:
                print(f"    Sent {idx + 1}/{len(test_cases)}")


def compute_metrics(log_file):
    total = 0
    rejected = 0
    deep_anomaly = 0
    with open(log_file, 'r') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            total += 1
            status = parts[3]
            response = parts[4]
            if status != "SUCCESS":
                rejected += 1
            else:
                if is_modbus_exception(response):
                    rejected += 1
                    deep_anomaly += 1
    tirr = (rejected / total) * 100 if total > 0 else 100.0
    aovd = (deep_anomaly / total) * 100 if total > 0 else 0.0
    return tirr, aovd, total, rejected, deep_anomaly


def main():
    print("=" * 60)
    print("  Step 4: Fuzzing & Comparison")
    print("=" * 60)
    print(f"  Target: {TARGET_IP}:{TARGET_PORT}")

    results = {}

    # --- Random Baseline ---
    print("\n[1/3] Random Baseline...")
    random_cases = generate_random_cases(NUM_BASELINE)
    random_file = os.path.join(TC_DIR, "random_cases.txt")
    with open(random_file, 'w') as f:
        for c in random_cases:
            f.write(c + '\n')
    log_random = os.path.join(LOG_DIR, "random.log")
    run_fuzzing(random_cases, TARGET_IP, TARGET_PORT, TEST_TIMEOUT, log_random)
    tirr_r, aovd_r, total_r, rej_r, anom_r = compute_metrics(log_random)
    results["Random"] = {"TIRR": tirr_r, "AoVD": aovd_r, "total": total_r, "anomaly": anom_r}
    print(f"  Random: total={total_r}, anomaly={anom_r}, TIRR={tirr_r:.2f}, AoVD={aovd_r:.2f}")

    # --- Mutation Baseline ---
    print("\n[2/3] Mutation Baseline...")
    mutation_cases = generate_mutation_cases(RAW_MSG_FILE, NUM_BASELINE)
    mutation_file = os.path.join(TC_DIR, "mutation_cases.txt")
    with open(mutation_file, 'w') as f:
        for c in mutation_cases:
            f.write(c + '\n')
    log_mut = os.path.join(LOG_DIR, "mutation.log")
    run_fuzzing(mutation_cases, TARGET_IP, TARGET_PORT, TEST_TIMEOUT, log_mut)
    tirr_m, aovd_m, total_m, rej_m, anom_m = compute_metrics(log_mut)
    results["Mutation"] = {"TIRR": tirr_m, "AoVD": aovd_m, "total": total_m, "anomaly": anom_m}
    print(f"  Mutation: total={total_m}, anomaly={anom_m}, TIRR={tirr_m:.2f}, AoVD={aovd_m:.2f}")

    # --- GAN (all, no filter) ---
    print("\n[3/3] GAN (all, no FC filter)...")
    gan_file = os.path.join(TC_DIR, "gan_all.txt")
    if not os.path.exists(gan_file):
        print(f"  GAN test cases not found: {gan_file}")
        print("  Skipping GAN test.")
    else:
        gan_cases = load_text(gan_file)
        log_gan = os.path.join(LOG_DIR, "gan.log")
        run_fuzzing(gan_cases, TARGET_IP, TARGET_PORT, TEST_TIMEOUT, log_gan)
        tirr_g, aovd_g, total_g, rej_g, anom_g = compute_metrics(log_gan)
        results["GAN_All"] = {"TIRR": tirr_g, "AoVD": aovd_g, "total": total_g, "anomaly": anom_g}
        print(f"  GAN(All): total={total_g}, anomaly={anom_g}, TIRR={tirr_g:.2f}, AoVD={aovd_g:.2f}")

    # --- GAN (valid, FC filtered) ---
    print("\n[3b/3] GAN (valid, FC filtered)...")
    gan_valid_file = os.path.join(TC_DIR, "gan_valid.txt")
    if os.path.exists(gan_valid_file):
        gan_valid_cases = load_text(gan_valid_file)
        if gan_valid_cases:
            log_gan_v = os.path.join(LOG_DIR, "gan_valid.log")
            run_fuzzing(gan_valid_cases, TARGET_IP, TARGET_PORT, TEST_TIMEOUT, log_gan_v)
            tirr_gv, aovd_gv, total_gv, rej_gv, anom_gv = compute_metrics(log_gan_v)
            results["GAN_Valid"] = {"TIRR": tirr_gv, "AoVD": aovd_gv, "total": total_gv, "anomaly": anom_gv}
            print(f"  GAN(Valid): total={total_gv}, anomaly={anom_gv}, TIRR={tirr_gv:.2f}, AoVD={aovd_gv:.2f}")

    # --- Save report ---
    report_file = os.path.join(REPORT_DIR, "comparison.json")
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("  COMPARISON RESULTS")
    print("=" * 60)
    print(f"  {'Method':<15} {'Total':>6} {'Anomaly':>8} {'TIRR':>10} {'AoVD':>10}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*10} {'-'*10}")
    for name, r in results.items():
        print(f"  {name:<15} {r['total']:>6} {r['anomaly']:>8} {r['TIRR']:>9.2f}% {r['AoVD']:>9.2f}")
    print(f"\n  Report saved to {report_file}")

    return results


if __name__ == "__main__":
    main()
