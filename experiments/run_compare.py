#原版

# experiments/run_compare.py
import os
import sys

import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from baseline.random_generator import generate_random_cases
from baseline.mutation_generator import generate_mutation_cases
from fuzzer.fuzzer_main import run_fuzzing
from evaluation.metrics import compute_tirr_aovd
from config import TEST_CASES_DIR, LOG_DIR, REPORT_DIR, NUM_BASELINE_CASES, RAW_MSG_FILE

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. 随机生成
    random_cases = generate_random_cases(num=NUM_BASELINE_CASES)
    random_file = os.path.join(TEST_CASES_DIR, "random_cases.txt")
    with open(random_file, 'w') as f:
        for case in random_cases:
            f.write(case + '\n')
    log_random = os.path.join(LOG_DIR, "baseline_random.log")
    run_fuzzing(random_file, log_random)
    tirr_rand, aovd_rand = compute_tirr_aovd(log_random)

    # 2. 随机变异
    mutation_cases = generate_mutation_cases(seed_file=RAW_MSG_FILE, num=NUM_BASELINE_CASES)
    mutation_file = os.path.join(TEST_CASES_DIR, "mutation_cases.txt")
    with open(mutation_file, 'w') as f:
        for case in mutation_cases:
            f.write(case + '\n')
    log_mut = os.path.join(LOG_DIR, "baseline_mutation.log")
    run_fuzzing(mutation_file, log_mut)
    tirr_mut, aovd_mut = compute_tirr_aovd(log_mut)

    # 3. GAN 测试用例
    gan_file = os.path.join(TEST_CASES_DIR, "gan_cases.txt")
    if not os.path.exists(gan_file):
        print("GAN test cases not found. Please run model/generate_cases.py first.")
        return
    log_gan = os.path.join(LOG_DIR, "fuzz_gan.log")
    run_fuzzing(gan_file, log_gan)
    tirr_gan, aovd_gan = compute_tirr_aovd(log_gan)

    # 保存结果
    results = {
        "Random": {"TIRR": tirr_rand, "AoVD": aovd_rand},
        "Mutation": {"TIRR": tirr_mut, "AoVD": aovd_mut},
        "GAN": {"TIRR": tirr_gan, "AoVD": aovd_gan}
    }
    report_file = os.path.join(REPORT_DIR, "comparison.json")
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {report_file}")
    print("TIRR (lower better):")
    for k, v in results.items():
        print(f"  {k}: {v['TIRR']:.2f}%")
    print("\nAoVD (higher better):")
    for k, v in results.items():
        print(f"  {k}: {v['AoVD']:.2f}")

if __name__ == "__main__":
    main()