# baseline/mutation_generator.py
import random
from utils.file_utils import load_text
from config import RAW_MSG_FILE

def mutate_byte(hex_msg, num_flips=1):
    data = bytearray.fromhex(hex_msg)
    if len(data) == 0:
        return hex_msg
    for _ in range(num_flips):
        pos = random.randint(0, len(data)-1)
        data[pos] ^= random.randint(1,255)
    return data.hex()

def generate_mutation_cases(seed_file=None, num=2000, seeds_per_mutation=20):
    if seed_file is None:
        seed_file = RAW_MSG_FILE
    seeds = load_text(seed_file)
    if not seeds:
        # 如果种子文件不存在，使用一个默认的 Modbus 请求作为种子
        seeds = ["00010000000601030000000A"] * 100
    cases = []
    while len(cases) < num:
        for seed in seeds:
            for _ in range(seeds_per_mutation):
                cases.append(mutate_byte(seed, random.randint(1,3)))
                if len(cases) >= num:
                    break
            if len(cases) >= num:
                break
    return cases[:num]