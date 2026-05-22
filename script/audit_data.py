import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR
from collections import Counter

raw_file = os.path.join(DATA_DIR, 'raw_messages.txt')
merged_file = os.path.join(DATA_DIR, 'merged_raw_messages.txt')

for label, path in [('raw_messages', raw_file), ('merged_raw_messages', merged_file)]:
    if not os.path.exists(path):
        print(f'{label}: NOT FOUND')
        continue
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    print(f'\n=== {label} === ({len(lines)} lines)')

    lens = Counter()
    funcs = Counter()
    bad = 0
    short = 0
    duplicate = len(lines) - len(set(lines))

    for msg in lines:
        blen = len(msg) // 2
        lens[blen] += 1
        if len(msg) < 14:
            short += 1
            bad += 1
            continue
        try:
            fc = int(msg[14:16], 16)
            if fc in (1,2,3,4,5,6,15,16):
                funcs[fc] += 1
            else:
                bad += 1
        except:
            bad += 1

    print(f'  重复行: {duplicate}')
    print(f'  长度不足(<14 hex): {short}')
    print(f'  格式异常(非标准功能码): {bad}')
    print(f'  功能码分布: {dict(sorted(funcs.items()))}')
    # 前15个最长的长度
    top_lens = sorted(lens.items(), key=lambda x: -x[1])[:10]
    print(f'  最常见的10个长度: {top_lens}')
    print(f'  长度范围: {min(lens.keys())}-{max(lens.keys())}')
    total_func_samples = sum(funcs.values())
    if total_func_samples > 0:
        print(f'  功能码占比: { {k: f"{v/total_func_samples*100:.1f}%" for k,v in sorted(funcs.items())} }')

    print(f'  样本:')
    for m in lines[:5]:
        print(f'    {m}')
