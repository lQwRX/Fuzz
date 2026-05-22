import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import struct
from collections import Counter, defaultdict
from config import DATA_DIR, RAW_MSG_FILE, MAX_MSG_LEN, MIN_MSG_LEN
from capture.message_collector import MessageCollector

random.seed(42)

MODBUS_FC = {1, 2, 3, 4, 5, 6, 15, 16}


def parse_modbus_tcp(msg_hex):
    if len(msg_hex) < 14:
        return None
    try:
        b = bytes.fromhex(msg_hex)
    except ValueError:
        return None
    if len(b) < 7:
        return None
    tid = struct.unpack('>H', b[0:2])[0]
    pid = struct.unpack('>H', b[2:4])[0]
    length_val = struct.unpack('>H', b[4:6])[0]
    uid = b[6]
    fc = b[7] if len(b) > 7 else None
    pdu_len = len(b) - 7
    return {
        'hex': msg_hex, 'byte_len': len(b),
        'tid': tid, 'pid': pid, 'length_val': length_val, 'uid': uid,
        'fc': fc, 'pdu_len': pdu_len, 'pdu_raw': b[7:].hex() if len(b) > 7 else ''
    }


def load_messages(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        return [l.strip() for l in f if l.strip()]


def save_messages(messages, filepath):
    with open(filepath, 'w') as f:
        for m in messages:
            f.write(m + '\n')
    print(f'  Saved {len(messages)} messages to {filepath}')


def classify_messages(messages):
    valid_requests = []
    invalid = []
    fc_stats = Counter()
    for m in messages:
        parsed = parse_modbus_tcp(m)
        if parsed is None:
            invalid.append(m)
            continue
        if parsed['fc'] is None or parsed['fc'] not in MODBUS_FC:
            invalid.append(m)
            continue
        if not (MIN_MSG_LEN <= parsed['byte_len'] <= MAX_MSG_LEN):
            invalid.append(m)
            continue
        valid_requests.append(m)
        fc_stats[parsed['fc']] += 1
    return valid_requests, invalid, fc_stats


def augment_fc03(real_fc03_samples, target_count):
    variants = []
    for _ in range(target_count):
        template = random.choice(real_fc03_samples)
        parsed = parse_modbus_tcp(template)
        if parsed is None:
            continue
        new_tid = random.randint(0, 65535)
        new_start = random.randint(0, 500)
        new_quantity = random.randint(1, 125)
        new_uid = random.choice([1, 255])
        header = struct.pack('>H H H B', new_tid, 0, 6, new_uid)
        pdu = struct.pack('>B H H', 3, new_start, new_quantity)
        msg = header + pdu
        variants.append(msg.hex())
    return variants


def smart_dedup(messages, keep_per_group=50):
    groups = defaultdict(list)
    for m in messages:
        parsed = parse_modbus_tcp(m)
        if parsed is None:
            key = ('invalid', len(m) // 2)
        else:
            key = (parsed['fc'], parsed['byte_len'])
        groups[key].append(m)

    result = []
    for key, msgs in groups.items():
        unique = list(dict.fromkeys(msgs))
        kept = random.sample(unique, min(keep_per_group, len(unique)))
        result.extend(kept)
    return result


def stratified_balance(messages, target_per_fc_len=None):
    groups = defaultdict(list)
    for m in messages:
        parsed = parse_modbus_tcp(m)
        if parsed is None:
            key = ('invalid', len(m) // 2)
        else:
            key = (parsed['fc'], parsed['byte_len'])
        groups[key].append(m)

    balanced = []
    min_count = 5
    max_group = 1000
    max_short_group = 400
    for key, msgs in groups.items():
        unique = list(dict.fromkeys(msgs))
        fc = key[0]
        blen = key[1] if isinstance(key[1], int) else MAX_MSG_LEN
        if fc in (15, 16):
            keep = min(len(unique), max_group * 2)
        elif fc in (1, 2, 3, 4):
            if blen <= 12:
                keep = min(len(unique), max_short_group)
            else:
                keep = min(len(unique), max_group)
        elif fc in (5, 6):
            keep = min(len(unique), max_short_group)
        else:
            keep = min(len(unique), max_group)
        keep = min(len(unique), max(keep, min_count))
        balanced.extend(random.sample(unique, keep))
    return balanced


def analyze_quality(label, messages):
    parsed = [parse_modbus_tcp(m) for m in messages]
    valid = [p for p in parsed if p is not None and p['fc'] is not None and p['fc'] in MODBUS_FC]
    invalid_count = len(parsed) - len(valid)
    duplicates = len(messages) - len(set(messages))

    lens = Counter(p['byte_len'] for p in valid)
    fcs = Counter(p['fc'] for p in valid)

    print(f'\n--- {label} ---')
    print(f'  Total: {len(messages)}')
    print(f'  Invalid: {invalid_count}')
    print(f'  Duplicates: {duplicates}')
    print(f'  Unique valid: {len(set(m["hex"] for m in valid) if valid else set())}')
    print(f'  FC distribution: {dict(sorted(fcs.items()))}')
    print(f'  Top 10 lengths: {sorted(lens.items(), key=lambda x: -x[1])[:10]}')
    if fcs:
        total = sum(fcs.values())
        for fc, cnt in sorted(fcs.items()):
            print(f'    FC{fc:02d}: {cnt:5d} ({cnt/total*100:5.1f}%)')


def improve_data_quality():
    print('=' * 60)
    print('DATA QUALITY IMPROVEMENT')
    print('=' * 60)

    raw_msgs = load_messages(RAW_MSG_FILE)
    print(f'\nLoaded {len(raw_msgs)} raw messages')

    merged_msgs = load_messages(os.path.join(DATA_DIR, 'merged_raw_messages.txt'))
    print(f'Loaded {len(merged_msgs)} merged messages')

    print('\n[Step 1] Classification (valid vs invalid)')
    raw_valid, raw_invalid, raw_fc_stats = classify_messages(raw_msgs)
    rmv_msg_file = os.path.join(DATA_DIR, 'raw_messages_cleaned.txt')
    invalid_file = os.path.join(DATA_DIR, 'raw_messages_invalid.txt')
    save_messages(raw_valid, rmv_msg_file)
    save_messages(raw_invalid, invalid_file)

    print(f'  Valid Modbus requests: {len(raw_valid)}')
    print(f'  Invalid/discarded:     {len(raw_invalid)}')
    print(f'  FC distribution: {dict(sorted(raw_fc_stats.items()))}')

    print('\n[Step 2] FC03 Augmentation')
    real_fc03 = [m for m in raw_valid if len(m) >= 16 and m[14:16] == '03']
    print(f'  Real FC03 samples: {len(real_fc03)}')
    if real_fc03:
        print(f'  Samples:')
        for s in real_fc03[:3]:
            print(f'    {s}')

    target_fc03 = 400
    aug_fc03 = augment_fc03(real_fc03, target_fc03)
    aug_fc03_unique = list(dict.fromkeys(aug_fc03))
    print(f'  Generated {len(aug_fc03)} FC03 variants, unique={len(aug_fc03_unique)}')

    aug_fc03_file = os.path.join(DATA_DIR, 'augmented_fc03.txt')
    save_messages(aug_fc03_unique, aug_fc03_file)

    print('\n[Step 3] Smart Deduplication')
    msg_with_fc03 = raw_valid + list(aug_fc03_unique)
    print(f'  Before dedup: {len(msg_with_fc03)}')
    deduped = smart_dedup(msg_with_fc03, keep_per_group=80)
    print(f'  After dedup:  {len(deduped)}')
    analyze_quality('After Dedup', deduped)

    print('\n[Step 4] Synthetic Data Enhancement')
    synthetic_count = 18000
    print(f'  Generating {synthetic_count} enhanced synthetic messages...')
    enhanced_synthetic = MessageCollector.generate_synthetic_modbus(synthetic_count, MAX_MSG_LEN)
    synthetic_deduped = list(dict.fromkeys(enhanced_synthetic))
    print(f'  Synthetic unique: {len(synthetic_deduped)}')
    analyze_quality('Synthetic Enhanced', synthetic_deduped)

    print('\n[Step 5] Stratified Balance (Real + FC03 Aug)')
    balanced_real = stratified_balance(deduped)
    print(f'  Balanced real+aug: {len(balanced_real)}')
    analyze_quality('Balanced Real', balanced_real)

    print('\n[Step 6] Merge & Final Balance')
    combined = balanced_real + synthetic_deduped
    print(f'  Combined before balance: {len(combined)}')
    final = stratified_balance(combined)
    analyze_quality('Final (merged & balanced)', final)

    final_file = os.path.join(DATA_DIR, 'improved_raw_messages.txt')
    save_messages(final, final_file)

    print('\n[Summary]')
    print(f'  Original raw:         {len(raw_msgs)} msgs')
    print(f'  Original merged:      {len(merged_msgs)} msgs')
    print(f'  Improved final:       {len(final)} msgs')
    print(f'  Output:               {final_file}')

    print('\nDone! Replace raw_messages.txt with improved version for training.')
    print('Or point preprocessing to use improved_raw_messages.txt')


if __name__ == '__main__':
    improve_data_quality()
