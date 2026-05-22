# scripts/boost_anomaly.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import BASE_DIR


def boost_anomaly(hex_str: str) -> list:
    """对合法 Modbus 请求进行异常增强，生成多个变体"""
    data = bytearray.fromhex(hex_str)
    if len(data) < 10:
        return [hex_str]
    func_code = data[7]
    variants = []
    # 原始用例保留
    variants.append(hex_str)
    # 1. 将起始地址改为 0xFFFF（高地址）
    if len(data) >= 10:
        data2 = data[:]
        data2[8] = 0xFF
        data2[9] = 0xFF
        variants.append(data2.hex())
    # 2. 将起始地址改为 0x0000
    if len(data) >= 10:
        data3 = data[:]
        data3[8] = 0x00
        data3[9] = 0x00
        variants.append(data3.hex())
    # 3. 对于读请求，将数量改为 0 或 125（边界）
    if func_code in (1, 2, 3, 4) and len(data) >= 12:
        # 数量 0
        data4 = data[:]
        data4[10] = 0x00
        data4[11] = 0x00
        variants.append(data4.hex())
        # 数量 125
        data5 = data[:]
        data5[10] = 0x00
        data5[11] = 125
        variants.append(data5.hex())
    # 4. 对于写多寄存器，将数据长度改为 0（实际不可能，但可尝试）
    # 这里简单起见，只做地址和数量
    return variants


def main():
    src_file = os.path.join(BASE_DIR, "valid_output", "test_cases", "gan_cases.txt")
    if not os.path.exists(src_file):
        print(f"Source file not found: {src_file}")
        return
    with open(src_file, 'r') as f:
        cases = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(cases)} valid test cases")

    boosted = []
    for case in cases:
        boosted.extend(boost_anomaly(case))
    boosted = list(dict.fromkeys(boosted))
    print(f"After anomaly boosting: {len(boosted)} cases")

    out_file = os.path.join(BASE_DIR, "valid_output", "test_cases", "gan_cases_anomaly.txt")
    with open(out_file, 'w') as f:
        for case in boosted:
            f.write(case + '\n')
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()