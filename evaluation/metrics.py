# evaluation/metrics.py
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

def compute_tirr_aovd(log_file):
    total = 0
    network_rejected = 0  # 网络层拒绝（超时/连接拒绝——报文根本发不过去）
    deep_anomaly = 0      # 协议层异常（Modbus 异常响应 0x80+——报文已送达但触发异常）
    with open(log_file, 'r') as f:
        lines = f.readlines()[1:]   # skip header
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            total += 1
            status = parts[3]
            response = parts[4]
            if status != "SUCCESS":
                network_rejected += 1
            else:
                if is_modbus_exception(response):
                    deep_anomaly += 1
    tirr = (network_rejected / total) * 100 if total > 0 else 100.0
    aovd = (deep_anomaly / total) * 100 if total > 0 else 0.0
    return tirr, aovd