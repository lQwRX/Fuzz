# feedback/extract_anomalies.py
import os


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
    # 可根据需要修改日志文件路径，例如从变异基线中提取
    log_file = "valid_output/logs/baseline_mutation.log"
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    anomalies = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        if len(lines) <= 1:
            print("Log file empty or no header")
            return
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            status = parts[3]
            response = parts[4]
            hex_msg = parts[2]
            if status == "SUCCESS" and is_modbus_exception(response):
                anomalies.append(hex_msg)

    anomalies = list(dict.fromkeys(anomalies))
    out_file = "valid_output/anomaly_seeds.txt"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        for msg in anomalies:
            f.write(msg + '\n')
    print(f"Extracted {len(anomalies)} anomaly seeds to {out_file}")


if __name__ == "__main__":
    main()