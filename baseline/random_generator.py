# baseline/random_generator.py
import random

def generate_random_cases(num=2000, min_len=7, max_len=100):
    cases = []
    for _ in range(num):
        length = random.randint(min_len, max_len)
        raw = bytes([random.randint(0,255) for _ in range(length)])
        cases.append(raw.hex())
    return cases