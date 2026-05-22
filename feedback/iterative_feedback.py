import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fuzzer.sender import ModbusSender
from fuzzer.logger import FuzzLogger
from evaluation.metrics import compute_tirr_aovd, is_modbus_exception
from model.generator import Generator
from config import (
    BASE_DIR, CLUSTER_DIR, TARGET_IP, TARGET_PORT, TEST_TIMEOUT,
    NUM_TARGET_CASES_PER_CLASS, DEVICE, EMBED_DIM, GEN_HIDDEN_DIM
)


class MessageDataset(Dataset):
    def __init__(self, data):
        self.data = torch.LongTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def is_valid_modbus_request(hex_str):
    if len(hex_str) < 14:
        return False
    try:
        func_code = int(hex_str[14:16], 16)
        return func_code in (1, 2, 3, 4, 5, 6, 15, 16)
    except:
        return False


def mutate_hex(hex_msg, num_flips=1):
    data = bytearray.fromhex(hex_msg)
    if not data:
        return hex_msg
    for _ in range(num_flips):
        pos = random.randint(0, len(data) - 1)
        data[pos] ^= random.randint(1, 255)
    return data.hex()


def hex_to_sequence(hex_msg, vocab, pad_idx, max_len):
    byte_data = bytes.fromhex(hex_msg)
    seq = [vocab.get(b, pad_idx) for b in byte_data]
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq += [pad_idx] * (max_len - len(seq))
    return seq


def load_cluster_data(cluster_type="no_clustering", class_id=0):
    data_path = os.path.join(CLUSTER_DIR, cluster_type)
    if cluster_type != "no_clustering":
        data_path = os.path.join(data_path, f"class_{class_id}")

    data = np.load(os.path.join(data_path, "data.npy"))
    with open(os.path.join(data_path, "vocab.json"), 'r') as f:
        vocab = json.load(f)
        vocab = {int(k) if k.isdigit() else k: v for k, v in vocab.items()}
    max_len = int(open(os.path.join(data_path, "max_len.txt")).read().strip())
    pad_idx = vocab['<PAD>']
    max_idx = data.max()
    vocab_size = max_idx + 1

    return data, vocab, max_len, pad_idx, vocab_size, data_path


def load_generator(vocab_size, max_len, pad_idx, model_path):
    gen = Generator(vocab_size, EMBED_DIM, GEN_HIDDEN_DIM, max_len, pad_idx)
    gen.load_state_dict(torch.load(model_path, map_location=DEVICE))
    gen.to(DEVICE)
    return gen


def generate_test_cases(gen, vocab, pad_idx, max_len, num_samples, start_token=2):
    gen.eval()
    inv_vocab = {v: k for k, v in vocab.items() if isinstance(k, int)}
    samples = gen.sample(num_samples, start_token, DEVICE, max_len)

    valid_cases = []
    all_cases = []
    for seq in samples:
        byte_seq = []
        for idx in seq:
            if idx == pad_idx:
                break
            if idx in inv_vocab:
                byte_seq.append(inv_vocab[idx])
        if byte_seq:
            hex_msg = bytes(byte_seq).hex()
            all_cases.append(hex_msg)
            if is_valid_modbus_request(hex_msg):
                valid_cases.append(hex_msg)

    valid_cases = list(dict.fromkeys(valid_cases))
    all_cases = list(dict.fromkeys(all_cases))
    return valid_cases, all_cases


def run_fuzzing_session(test_cases, target_ip, target_port, timeout, log_file):
    sender = ModbusSender(target_ip, target_port, timeout)
    logger = FuzzLogger(log_file)
    for idx, hex_msg in enumerate(test_cases):
        success, resp = sender.send_hex(hex_msg)
        status = "SUCCESS" if success else "TIMEOUT/REFUSED"
        logger.log(idx, hex_msg, status, resp)
        if (idx + 1) % 500 == 0:
            print(f"    Sent {idx + 1}/{len(test_cases)}")
    logger.close()
    return log_file


def extract_anomaly_seeds(log_file):
    anomalies = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            status = parts[3]
            response = parts[4]
            hex_msg = parts[2]
            if status == "SUCCESS" and is_modbus_exception(response):
                anomalies.append(hex_msg)
    return list(dict.fromkeys(anomalies))


def retrain_generator(gen, original_data, anomaly_seeds, vocab, pad_idx, max_len,
                      epochs=3, lr=1e-4, mutations_per_seed=8, anomaly_weight=8.0):
    expanded = []
    for seed in anomaly_seeds[:500]:
        expanded.append(seed)
        for _ in range(mutations_per_seed):
            mutated = mutate_hex(seed, random.randint(1, 3))
            if len(mutated) // 2 > max_len:
                mutated = mutated[:max_len * 2]
            expanded.append(mutated)
    expanded = list(dict.fromkeys(expanded))

    new_seqs = [hex_to_sequence(h, vocab, pad_idx, max_len) for h in expanded]
    new_data = np.array(new_seqs, dtype=np.int64)
    combined_data = np.vstack([original_data, new_data])

    normal_count = original_data.shape[0]
    anomaly_count = new_data.shape[0]

    sample_weights = []
    for i in range(combined_data.shape[0]):
        if i < normal_count:
            sample_weights.append(1.0)
        else:
            sample_weights.append(anomaly_weight)

    dataset = MessageDataset(combined_data)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=int(combined_data.shape[0] * 1.2),
        replacement=True
    )
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)
    optimizer = torch.optim.Adam(gen.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    gen.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits, _ = gen(inputs)
            ce_loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad()
            ce_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
            optimizer.step()
            total_loss += ce_loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"    Epoch {epoch + 1}/{epochs} loss: {avg_loss:.4f}")

    return gen


def _discover_all_models(base_model_step=10):
    all_models = []
    for ct in ["no_clustering", "same_length", "advanced"]:
        ct_model_dir = os.path.join(BASE_DIR, "model", "checkpoints_improved", ct)
        if not os.path.exists(ct_model_dir):
            continue
        for item in sorted(os.listdir(ct_model_dir)):
            if not item.startswith("class_"):
                continue
            cid = int(item.split("_")[1])
            model_path = os.path.join(ct_model_dir, item, f"generator_step{base_model_step}.pth")
            if not os.path.exists(model_path):
                continue
            data_path = os.path.join(CLUSTER_DIR, ct)
            if ct != "no_clustering":
                data_path = os.path.join(data_path, item)
            if not os.path.exists(os.path.join(data_path, "data.npy")):
                continue
            all_models.append({
                "cluster_type": ct,
                "class_id": cid,
                "model_path": model_path,
                "data_path": data_path
            })
    return all_models


def _generate_from_all_models(model_entries, num_per_model, base_model_step=10):
    all_valid = []
    all_total = []
    per_model_stats = []
    for entry in model_entries:
        ct, cid = entry["cluster_type"], entry["class_id"]
        data_path = entry["data_path"]
        model_path = entry["model_path"]

        data = np.load(os.path.join(data_path, "data.npy"))
        with open(os.path.join(data_path, "vocab.json"), 'r') as f:
            vocab = json.load(f)
            vocab = {int(k) if k.isdigit() else k: v for k, v in vocab.items()}
        max_len = int(open(os.path.join(data_path, "max_len.txt")).read().strip())
        pad_idx = vocab['<PAD>']
        vocab_size = max(vocab.values()) + 1

        gen = load_generator(vocab_size, max_len, pad_idx, model_path)
        valid, total = generate_test_cases(gen, vocab, pad_idx, max_len, num_per_model)
        all_valid.extend(valid)
        all_total.extend(total)
        per_model_stats.append({
            "cluster_type": ct, "class_id": cid,
            "valid": len(valid), "total": len(total)
        })
        print(f"    {ct}/class_{cid}: {len(valid)} 有效 / {len(total)} 总计")

    all_valid = list(dict.fromkeys(all_valid))
    all_total = list(dict.fromkeys(all_total))
    return all_valid, all_total, per_model_stats


def iterative_feedback_loop(
        num_iterations=3,
        num_generate=2000,
        cluster_type="no_clustering",
        class_id=0,
        retrain_epochs=3,
        retrain_lr=1e-4,
        mutations_per_seed=8,
        target_ip=TARGET_IP,
        target_port=TARGET_PORT,
        timeout=TEST_TIMEOUT,
        base_model_step=10,
        output_dir=None
):
    is_mixed = (cluster_type == "mixed")

    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "iterative_feedback_output")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("  迭代反馈调节 (Iterative Feedback Loop)")
    print("=" * 60)
    print(f"  模式: {'混合聚类 (Mixed)' if is_mixed else f'{cluster_type}/class_{class_id}'}")
    print(f"  迭代次数: {num_iterations}")
    print(f"  每轮生成: {num_generate}")
    print(f"  微调轮数: {retrain_epochs}")
    print(f"  目标: {target_ip}:{target_port}")
    print(f"  输出目录: {run_dir}")
    print("=" * 60)

    if is_mixed:
        model_entries = _discover_all_models(base_model_step)
        if not model_entries:
            print("未找到任何可用模型")
            return None
        print(f"\n发现 {len(model_entries)} 个模型:")
        for e in model_entries:
            print(f"  {e['cluster_type']}/class_{e['class_id']}")

        retrain_entry = None
        for e in model_entries:
            if e["cluster_type"] == "no_clustering":
                retrain_entry = e
                break
        if retrain_entry is None:
            retrain_entry = model_entries[0]
        print(f"微调目标模型: {retrain_entry['cluster_type']}/class_{retrain_entry['class_id']}")

        data, vocab, max_len, pad_idx, vocab_size, _ = load_cluster_data(
            retrain_entry["cluster_type"], retrain_entry["class_id"]
        )
        gen = load_generator(vocab_size, max_len, pad_idx, retrain_entry["model_path"])
        print(f"数据加载完成: shape={data.shape}, vocab_size={vocab_size}, max_len={max_len}")
    else:
        data, vocab, max_len, pad_idx, vocab_size, data_path = load_cluster_data(cluster_type, class_id)
        print(f"\n数据加载完成: shape={data.shape}, vocab_size={vocab_size}, max_len={max_len}")

        model_dir = os.path.join(BASE_DIR, "model", "checkpoints_improved", cluster_type, f"class_{class_id}")
        initial_model = os.path.join(model_dir, f"generator_step{base_model_step}.pth")
        if not os.path.exists(initial_model):
            print(f"初始模型不存在: {initial_model}")
            return None

        gen = load_generator(vocab_size, max_len, pad_idx, initial_model)
        print(f"初始模型加载完成: {initial_model}")
        model_entries = None
        retrain_entry = None

    history = []
    current_model_path = retrain_entry["model_path"] if is_mixed else initial_model
    cumulative_seeds = []
    prev_cumulative_count = 0

    for iteration in range(1, num_iterations + 1):
        iter_dir = os.path.join(run_dir, f"iter_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)

        print(f"\n{'=' * 50}")
        print(f"  迭代第 {iteration}/{num_iterations} 轮")
        print(f"{'=' * 50}")

        # Step 1: 生成测试用例
        if is_mixed:
            print(f"\n[Step 1] 混合生成测试用例 (每模型 {num_generate // len(model_entries)} 条)...")
            num_per = max(num_generate // len(model_entries), 100)
            valid_cases, all_cases, per_stats = _generate_from_all_models(
                model_entries, num_per, base_model_step
            )
            print(f"  混合汇总: {len(valid_cases)} 条有效 / {len(all_cases)} 条总计")
        else:
            print(f"\n[Step 1] 生成测试用例 ({num_generate} 条)...")
            valid_cases, all_cases = generate_test_cases(
                gen, vocab, pad_idx, max_len, num_generate
            )
            per_stats = None
            print(f"  生成完成: {len(valid_cases)} 条有效 / {len(all_cases)} 条总计")

        test_file = os.path.join(iter_dir, "test_cases.txt")
        with open(test_file, 'w') as f:
            for case in valid_cases:
                f.write(case + '\n')

        # Step 2: 执行模糊测试
        print(f"\n[Step 2] 执行模糊测试 (目标: {target_ip}:{target_port})...")
        log_file = os.path.join(iter_dir, "fuzz_log.txt")
        try:
            run_fuzzing_session(valid_cases, target_ip, target_port, timeout, log_file)
        except Exception as e:
            print(f"  模糊测试异常: {e}")
            print(f"  跳过本轮反馈，继续下一轮...")
            tirr, aovd = 100.0, 0.0
            anomaly_seeds = []
        else:
            # Step 3: 计算指标
            print(f"\n[Step 3] 计算评估指标...")
            tirr, aovd = compute_tirr_aovd(log_file)
            print(f"  TIRR: {tirr:.2f}% (越低越好)")
            print(f"  AoVD: {aovd:.2f} (越高越好)")

            # Step 4: 提取异常种子
            print(f"\n[Step 4] 提取异常种子...")
            anomaly_seeds = extract_anomaly_seeds(log_file)
            print(f"  提取到 {len(anomaly_seeds)} 条异常种子")

            seed_file = os.path.join(iter_dir, "anomaly_seeds.txt")
            with open(seed_file, 'w') as f:
                for seed in anomaly_seeds:
                    f.write(seed + '\n')

        cumulative_seeds.extend(anomaly_seeds)
        cumulative_seeds = list(dict.fromkeys(cumulative_seeds))

        iter_result = {
            "iteration": iteration,
            "valid_cases": len(valid_cases),
            "total_cases": len(all_cases),
            "anomaly_seeds": len(anomaly_seeds),
            "cumulative_seeds": len(cumulative_seeds),
            "tirr": tirr,
            "aovd": aovd,
            "model_path": current_model_path,
            "mode": "mixed" if is_mixed else f"{cluster_type}/class_{class_id}"
        }
        if per_stats:
            iter_result["per_model_stats"] = per_stats
        history.append(iter_result)

        print(f"\n  本轮结果: TIRR={tirr:.2f}%, AoVD={aovd:.2f}, "
              f"新种子={len(anomaly_seeds)}, 累计种子={len(cumulative_seeds)}")

        # Step 5: 反馈微调 (带早停检测)
        should_continue = True
        if iteration < num_iterations and cumulative_seeds:
            prev_aovd = history[-2]["aovd"] if len(history) >= 2 else 0
            prev_tirr = history[-2]["tirr"] if len(history) >= 2 else 100

            aovd_change = aovd - prev_aovd
            tirr_change = prev_tirr - tirr

            if len(history) >= 2:
                print(f"\n  变化: ΔTIRR={tirr_change:+.2f}%, ΔAoVD={aovd_change:+.2f}")
                if aovd_change < -2.0 and tirr_change < 2.0:
                    print(f"  ⚠️ AOVD下降且TIRR无改善，触发早停!")
                    should_continue = False

            if cumulative_seeds == prev_cumulative_count:
                print(f"  无新增异常种子，累积种子池已收敛")
                should_continue = False

            if should_continue:
                print(f"\n[Step 5] 反馈微调模型 (epochs={retrain_epochs}, "
                      f"seeds={len(cumulative_seeds)}, mutations={mutations_per_seed})...")
                if is_mixed:
                    print(f"  微调目标: {retrain_entry['cluster_type']}/class_{retrain_entry['class_id']}")
                gen = retrain_generator(
                    gen, data, cumulative_seeds, vocab, pad_idx, max_len,
                    epochs=retrain_epochs, lr=retrain_lr,
                    mutations_per_seed=mutations_per_seed,
                    anomaly_weight=8.0
                )

                current_model_path = os.path.join(iter_dir, "generator_retrained.pth")
                torch.save(gen.state_dict(), current_model_path)
                print(f"  微调完成，模型保存至: {current_model_path}")

                if is_mixed:
                    model_entries = _update_mixed_models(model_entries, gen, vocab, pad_idx, max_len, vocab_size)
            else:
                print(f"\n[Step 5] 跳过微调 (已达收敛)")
                break
        elif not cumulative_seeds:
            print(f"\n[Step 5] 无异常种子，跳过微调")
            should_continue = False
        else:
            print(f"\n[Step 5] 已达最终迭代，保存最终模型")
            final_model_path = os.path.join(run_dir, "generator_final.pth")
            torch.save(gen.state_dict(), final_model_path)
            print(f"  最终模型保存至: {final_model_path}")

        if not should_continue:
            if not os.path.exists(os.path.join(run_dir, "generator_final.pth")):
                torch.save(gen.state_dict(), os.path.join(run_dir, "generator_final.pth"))
            break

        prev_cumulative_count = len(cumulative_seeds)

    if not history[-1].get("early_stop_reason"):
        pass

    # 保存最终模型
    final_model_path = os.path.join(run_dir, "generator_final.pth")
    if not os.path.exists(final_model_path):
        torch.save(gen.state_dict(), final_model_path)

    # 用最终模型生成一批测试用例
    print(f"\n{'=' * 50}")
    print(f"  使用最终模型生成测试用例...")
    print(f"{'=' * 50}")
    if is_mixed:
        final_valid, final_all, _ = _generate_from_all_models(
            model_entries, max(num_generate // len(model_entries), 100), base_model_step
        )
    else:
        final_valid, final_all = generate_test_cases(
            gen, vocab, pad_idx, max_len, num_generate
        )
    final_test_file = os.path.join(run_dir, "final_test_cases.txt")
    with open(final_test_file, 'w') as f:
        for case in final_valid:
            f.write(case + '\n')
    print(f"  最终生成: {len(final_valid)} 条有效测试用例")

    # 保存历史记录
    history_file = os.path.join(run_dir, "iteration_history.json")
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print(f"\n{'=' * 60}")
    print(f"  迭代反馈调节完成!")
    print(f"{'=' * 60}")
    print(f"  模式: {'混合聚类' if is_mixed else cluster_type}")
    print(f"  迭代轮次: {num_iterations}")
    print(f"  最终模型: {final_model_path}")
    print(f"  最终测试用例: {final_test_file}")
    print(f"\n  各轮指标:")
    print(f"  {'轮次':>4} | {'TIRR(%)':>8} | {'AoVD':>8} | {'新种子':>6} | {'累计种子':>8}")
    print(f"  {'----':>4}-+-{'--------':>8}-+-{'--------':>8}-+-{'------':>6}-+-{'--------':>8}")
    for h in history:
        print(f"  {h['iteration']:>4} | {h['tirr']:>8.2f} | {h['aovd']:>8.2f} | "
              f"{h['anomaly_seeds']:>6} | {h['cumulative_seeds']:>8}")

    return {
        "run_dir": run_dir,
        "final_model": final_model_path,
        "final_test_cases": final_test_file,
        "history": history
    }


def _update_mixed_models(model_entries, retrained_gen, vocab, pad_idx, max_len, vocab_size):
    updated = []
    for entry in model_entries:
        ct, cid = entry["cluster_type"], entry["class_id"]
        data_path = entry["data_path"]

        data = np.load(os.path.join(data_path, "data.npy"))
        with open(os.path.join(data_path, "vocab.json"), 'r') as f:
            local_vocab = json.load(f)
            local_vocab = {int(k) if k.isdigit() else k: v for k, v in local_vocab.items()}
        local_max_len = int(open(os.path.join(data_path, "max_len.txt")).read().strip())
        local_pad_idx = local_vocab['<PAD>']
        local_vocab_size = max(local_vocab.values()) + 1

        local_gen = load_generator(local_vocab_size, local_max_len, local_pad_idx, entry["model_path"])

        try:
            local_gen.load_state_dict(retrained_gen.state_dict(), strict=False)
        except:
            pass

        updated.append(entry)

    return model_entries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="迭代反馈调节")
    parser.add_argument("--iterations", type=int, default=3, help="迭代轮数")
    parser.add_argument("--num_generate", type=int, default=2000, help="每轮生成数量")
    parser.add_argument("--cluster_type", default="no_clustering",
                        choices=["no_clustering", "same_length", "advanced", "mixed"],
                        help="聚类类型 (mixed=混合所有聚类)")
    parser.add_argument("--class_id", type=int, default=0, help="类别ID")
    parser.add_argument("--retrain_epochs", type=int, default=3, help="每轮微调epoch数")
    parser.add_argument("--retrain_lr", type=float, default=1e-4, help="微调学习率")
    parser.add_argument("--mutations_per_seed", type=int, default=8, help="每条种子变体数")
    parser.add_argument("--target_ip", default=TARGET_IP, help="目标IP")
    parser.add_argument("--target_port", type=int, default=TARGET_PORT, help="目标端口")
    args = parser.parse_args()

    result = iterative_feedback_loop(
        num_iterations=args.iterations,
        num_generate=args.num_generate,
        cluster_type=args.cluster_type,
        class_id=args.class_id,
        retrain_epochs=args.retrain_epochs,
        retrain_lr=args.retrain_lr,
        mutations_per_seed=args.mutations_per_seed,
        target_ip=args.target_ip,
        target_port=args.target_port
    )
