import os

# 使用绝对路径（请根据您的实际路径确认）
BASE = r"D:\PythonWorkSpace\Fuzz"
LOG_FILE = os.path.join(BASE, "valid_output", "logs", "baseline_mutation.log")
OUT_FILE = os.path.join(BASE, "valid_output", "anomaly_seeds.txt")

def is_modbus_exception(response_hex):
    if not response_hex or len(response_hex) < 8:
        return False
    try:
        resp_bytes = bytes.fromhex(response_hex)
        if len(resp_bytes) < 8:
            return False
        func_code = resp_bytes[7]
        return func_code >= 0x80
    except:
        return False

def main():
    if not os.path.exists(LOG_FILE):
        print(f"Log file not found: {LOG_FILE}")
        return

    anomalies = []
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
        if len(lines) <= 1:
            print("Log file has no data lines.")
            return
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            status = parts[3]
            response = parts[4]
            hex_msg = parts[2]
            # 只取成功发送且响应为 Modbus 异常（功能码 >= 0x80）
            if status == "SUCCESS" and is_modbus_exception(response):
                anomalies.append(hex_msg)

    anomalies = list(dict.fromkeys(anomalies))
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        for msg in anomalies:
            f.write(msg + '\n')
    print(f"Extracted {len(anomalies)} anomaly seeds from mutation baseline to {OUT_FILE}")

if __name__ == "__main__":
    main()