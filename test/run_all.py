import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    start = time.time()
    print("=" * 60)
    print("  TEST PIPELINE: Pretrain-Only Experiment")
    print("  Data: raw_messages.txt (no augmentation)")
    print("  Training: Pretrain Generator only (no adversarial)")
    print("  Output: test/output_test/")
    print("=" * 60)

    # Step 1
    print("\n" + "#" * 60)
    print("# STEP 1: Data Preprocessing")
    print("#" * 60)
    import preprocess
    preprocess.main()

    # Step 2
    print("\n" + "#" * 60)
    print("# STEP 2: Pretrain Generator")
    print("#" * 60)
    import pretrain_only
    pretrain_only.main()

    # Step 3
    print("\n" + "#" * 60)
    print("# STEP 3: Generate Test Cases")
    print("#" * 60)
    import generate
    generate.main()

    # Step 4
    print("\n" + "#" * 60)
    print("# STEP 4: Fuzzing & Comparison")
    print("#" * 60)
    import fuzz_and_compare
    fuzz_and_compare.main()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  ALL DONE in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
