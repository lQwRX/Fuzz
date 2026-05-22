import os, json, re
from collections import Counter, defaultdict

BASE = "valid_output_retrained"
LOGS = os.path.join(BASE, "logs")
REPORTS = os.path.join(BASE, "reports")
OUT = os.path.join(BASE, "paper_analysis.txt")

MODBUS_EXCEPTION_NAMES = {
    0x01: "Illegal Function",
    0x02: "Illegal Data Address",
    0x03: "Illegal Data Value",
    0x04: "Slave Device Failure",
    0x05: "Acknowledge",
    0x06: "Slave Device Busy",
    0x07: "Negative Acknowledge",
    0x08: "Memory Parity Error",
    0x0A: "Gateway Path Unavailable",
    0x0B: "Gateway Target Device Failed to Respond",
}

MODBUS_FUNCTION_NAMES = {
    0x01: "FC01 Read Coils",
    0x02: "FC02 Read Discrete Inputs",
    0x03: "FC03 Read Holding Registers",
    0x04: "FC04 Read Input Registers",
    0x05: "FC05 Write Single Coil",
    0x06: "FC06 Write Single Register",
    0x07: "FC07 Read Exception Status",
    0x08: "FC08 Diagnostics",
    0x0B: "FC11 Get Comm Event Counter",
    0x0C: "FC12 Get Comm Event Log",
    0x0F: "FC15 Write Multiple Coils",
    0x10: "FC16 Write Multiple Registers",
    0x11: "FC17 Report Server ID",
    0x14: "FC20 Read File Record",
    0x15: "FC21 Write File Record",
    0x16: "FC22 Mask Write Register",
    0x17: "FC23 Read/Write Multiple Registers",
    0x18: "FC24 Read FIFO Queue",
}


def parse_log(log_file):
    entries = []
    with open(log_file) as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            ts, idx, hex_msg, status, response = parts[0], int(parts[1]), parts[2], parts[3], parts[4]
            entries.append({"ts": ts, "idx": idx, "hex_msg": hex_msg, "status": status, "response": response})
    return entries


def parse_modbus_response(hex_str):
    if not hex_str or len(hex_str) < 16:
        return None, None
    raw = bytes.fromhex(hex_str)
    if len(raw) < 8:
        return None, None
    func_code = raw[7]
    exception_code = None
    if func_code >= 0x80:
        exception_code = raw[8] if len(raw) > 8 else None
    return func_code, exception_code


def get_func_code_from_msg(hex_str):
    if not hex_str or len(hex_str) < 16:
        return None
    raw = bytes.fromhex(hex_str)
    if len(raw) < 8:
        return None
    return raw[7]


def classify_failure(response):
    if "10054" in response:
        return "Connection Reset (10054)"
    elif "10060" in response or "timed out" in response.lower():
        return "Connection Timeout (10060)"
    elif "10061" in response:
        return "Connection Refused (10061)"
    elif "10048" in response:
        return "Address In Use (10048)"
    else:
        return f"Other ({response[:40]})"


def analyze():
    report = []
    report.append("=" * 70)
    report.append("PAPER EXPERIMENT ANALYSIS - valid_output_retrained")
    report.append("=" * 70)

    # Load comparison results
    cmp = json.load(open(os.path.join(REPORTS, "comparison.json")))
    report.append("\n" + "=" * 70)
    report.append("0. OVERALL RESULTS (from comparison.json)")
    report.append("=" * 70)
    report.append(f"  {'Method':<20} {'TIRR(%)':<15} {'AoVD(%)':<15}")
    report.append(f"  {'-'*20} {'-'*15} {'-'*15}")
    for method, metrics in cmp.items():
        report.append(f"  {method:<20} {metrics['TIRR']:<15.2f} {metrics['AoVD']:<15.2f}")

    # =============================================
    # 1. ANOMALY TYPE DISTRIBUTION
    # =============================================
    report.append("\n\n" + "=" * 70)
    report.append("1. ANOMALY TYPE DISTRIBUTION (异常类型分布)")
    report.append("=" * 70)

    for log_name in ["fuzz_gan.log", "baseline_mutation.log", "baseline_random.log"]:
        log_path = os.path.join(LOGS, log_name)
        if not os.path.exists(log_path):
            continue
        method = log_name.replace("baseline_", "").replace(".log", "").replace("fuzz_", "GAN ")
        method = method.replace("mutation", "Mutation").replace("random", "Random")
        if "GAN" in method:
            method = "GAN_Retrained"

        entries = parse_log(log_path)
        total = len(entries)
        success = [e for e in entries if e["status"] == "SUCCESS"]
        rejected = [e for e in entries if e["status"] != "SUCCESS"]
        exceptions = [e for e in success if e["response"] and parse_modbus_response(e["response"])[0] is not None and parse_modbus_response(e["response"])[0] >= 0x80]

        # Count exception types
        exc_counter = Counter()
        for e in exceptions:
            fc, exc = parse_modbus_response(e["response"])
            if exc is not None:
                exc_counter[exc] += 1

        total_exceptions = sum(exc_counter.values())
        report.append(f"\n{method} ({total} cases, {len(success)} success, {len(exceptions)} exceptions):")

        if total_exceptions > 0:
            report.append(f"  {'Exception Code':<20} {'Name':<35} {'Count':<8} {'% of Exceptions':<18}")
            report.append(f"  {'-'*20} {'-'*35} {'-'*8} {'-'*18}")
            for exc_code in sorted(exc_counter.keys()):
                name = MODBUS_EXCEPTION_NAMES.get(exc_code, f"Unknown (0x{exc_code:02X})")
                cnt = exc_counter[exc_code]
                pct = cnt / total_exceptions * 100
                report.append(f"  0x{exc_code:02X}{'':<18} {name:<35} {cnt:<8} {pct:<18.1f}")
        else:
            report.append("  (No Modbus exceptions triggered)")

        # Also summarize by exception function code
        exc_func_counter = Counter()
        for e in exceptions:
            fc, exc = parse_modbus_response(e["response"])
            exc_func_counter[fc] += 1

        if exc_func_counter:
            report.append(f"\n  By exception function code (orig func that caused the exception):")
            for fc in sorted(exc_func_counter.keys()):
                orig_func = fc - 0x80
                orig_name = MODBUS_FUNCTION_NAMES.get(orig_func, f"0x{orig_func:02X}")
                report.append(f"    0x{fc:02X} (orig {orig_name}): {exc_func_counter[fc]} cases")

    # =============================================
    # 2. FUNCTION CODE COVERAGE
    # =============================================
    report.append("\n\n" + "=" * 70)
    report.append("2. FUNCTION CODE COVERAGE (功能码覆盖分析)")
    report.append("=" * 70)

    # All known Modbus function codes
    all_fcs = set(range(0x01, 0x1A))  # FC01 - FC24
    all_fcs.update([0x56, 0x5A, 0x7E, 0x7F])  # Some extended codes

    for log_name in ["fuzz_gan.log", "baseline_mutation.log", "baseline_random.log"]:
        log_path = os.path.join(LOGS, log_name)
        if not os.path.exists(log_path):
            continue
        method = log_name.replace("baseline_", "").replace(".log", "").replace("fuzz_", "GAN ")
        method = method.replace("mutation", "Mutation").replace("random", "Random")
        if "GAN" in method:
            method = "GAN_Retrained"

        entries = parse_log(log_path)
        sent_fcs = Counter()
        success_fcs = Counter()
        exception_fcs = Counter()

        for e in entries:
            fc = get_func_code_from_msg(e["hex_msg"])
            if fc is not None:
                sent_fcs[fc] += 1
            if e["status"] == "SUCCESS" and e["response"]:
                rfc, _ = parse_modbus_response(e["response"])
                if rfc is not None:
                    success_fcs[rfc] += 1
                    if rfc >= 0x80:
                        exception_fcs[rfc] += 1

        unique_sent = len(sent_fcs)
        unique_valid = len([fc for fc in success_fcs if fc < 0x80])
        unique_exc = len(exception_fcs)

        report.append(f"\n{method} ({len(entries)} cases total):")
        report.append(f"  Unique function codes sent: {unique_sent}")
        report.append(f"  Unique function codes with valid response: {unique_valid}")
        report.append(f"  Unique function codes that triggered exceptions: {unique_exc}")

        # Top function codes by sent count
        report.append(f"\n  Top 15 function codes (by sent count):")
        report.append(f"  {'FC':<8} {'Name':<35} {'Sent':<8} {'Success':<10} {'Exception':<10}")
        report.append(f"  {'-'*8} {'-'*35} {'-'*8} {'-'*10} {'-'*10}")
        for fc, cnt in sent_fcs.most_common(15):
            name = MODBUS_FUNCTION_NAMES.get(fc, f"0x{fc:02X} (Unknown)")
            sc = success_fcs.get(fc, 0)
            ec = exception_fcs.get(fc + 0x80, 0)
            report.append(f"  0x{fc:02X}{'':<6} {name:<35} {cnt:<8} {sc:<10} {ec:<10}")

    # =============================================
    # 3. FIELD MUTATION DEPTH ANALYSIS
    # =============================================
    report.append("\n\n" + "=" * 70)
    report.append("3. FIELD MUTATION DEPTH ANALYSIS (字段变异深度分析)")
    report.append("=" * 70)

    for log_name in ["fuzz_gan.log", "baseline_mutation.log"]:
        log_path = os.path.join(LOGS, log_name)
        if not os.path.exists(log_path):
            continue
        method = log_name.replace("baseline_", "").replace(".log", "").replace("fuzz_", "GAN ")
        method = method.replace("mutation", "Mutation").replace("random", "Random")
        if "GAN" in method:
            method = "GAN_Retrained"

        entries = parse_log(log_path)
        success_entries = [e for e in entries if e["status"] == "SUCCESS"]

        # Analyze byte diversity in key Modbus TCP fields for sent messages
        field_values = defaultdict(list)
        for e in entries:
            raw = bytes.fromhex(e["hex_msg"])
            if len(raw) < 12:
                continue
            # Transaction ID (bytes 0-1)
            tid = raw[0] * 256 + raw[1]
            field_values["Transaction ID"].append(tid)
            # Protocol ID (bytes 2-3) - should be 0
            pid = raw[2] * 256 + raw[3]
            field_values["Protocol ID"].append(pid)
            # Length (bytes 4-5)
            leng = raw[4] * 256 + raw[5]
            field_values["Length"].append(leng)
            # Unit ID (byte 6)
            field_values["Unit ID"].append(raw[6])
            # Function Code (byte 7)
            field_values["Function Code"].append(raw[7])
            # First data byte (byte 8)
            if len(raw) > 8:
                field_values["Data Byte 1"].append(raw[8])
            if len(raw) > 12:
                field_values["Data Bytes 4-5"].append(raw[11]*256 + raw[12] if len(raw) > 12 else 0)

        report.append(f"\n{method} ({len(entries)} total, {len(success_entries)} success):")
        report.append(f"  {'Field':<20} {'Unique Values':<15} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<12}")
        report.append(f"  {'-'*20} {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
        for field in ["Transaction ID", "Protocol ID", "Length", "Unit ID", "Function Code", "Data Byte 1", "Data Bytes 4-5"]:
            if field not in field_values:
                continue
            vals = field_values[field]
            unique = len(set(vals))
            min_v = min(vals)
            max_v = max(vals)
            mean_v = sum(vals) / len(vals)
            from math import sqrt
            variance = sum((x - mean_v)**2 for x in vals) / len(vals)
            std_v = sqrt(variance)
            report.append(f"  {field:<20} {unique:<15} {min_v:<10} {max_v:<10} {mean_v:<10.1f} {std_v:<12.1f}")

    # =============================================
    # 4. FAILURE REASON CLASSIFICATION
    # =============================================
    report.append("\n\n" + "=" * 70)
    report.append("4. RECEPTION FAILURE REASON CLASSIFICATION (接收失败原因分类)")
    report.append("=" * 70)

    for log_name in ["fuzz_gan.log", "baseline_mutation.log", "baseline_random.log"]:
        log_path = os.path.join(LOGS, log_name)
        if not os.path.exists(log_path):
            continue
        method = log_name.replace("baseline_", "").replace(".log", "").replace("fuzz_", "GAN ")
        method = method.replace("mutation", "Mutation").replace("random", "Random")
        if "GAN" in method:
            method = "GAN_Retrained"

        entries = parse_log(log_path)
        total = len(entries)
        rejected_entries = [e for e in entries if e["status"] != "SUCCESS"]
        failure_counter = Counter()
        for e in rejected_entries:
            reason = classify_failure(e["response"])
            failure_counter[reason] += 1

        report.append(f"\n{method} (total: {total}, rejected: {len(rejected_entries)}, TIRR: {len(rejected_entries)/total*100:.2f}%):")
        if failure_counter:
            report.append(f"  {'Failure Reason':<40} {'Count':<8} {'% of Rejected':<18} {'% of Total':<12}")
            report.append(f"  {'-'*40} {'-'*8} {'-'*18} {'-'*12}")
            for reason, cnt in failure_counter.most_common():
                pct_rej = cnt / len(rejected_entries) * 100
                pct_tot = cnt / total * 100
                report.append(f"  {reason:<40} {cnt:<8} {pct_rej:<18.1f} {pct_tot:<12.1f}")

    # =============================================
    # 5. CASE STUDY
    # =============================================
    report.append("\n\n" + "=" * 70)
    report.append("5. CASE STUDY (案例展示)")
    report.append("=" * 70)

    for log_name in ["fuzz_gan.log", "baseline_mutation.log"]:
        log_path = os.path.join(LOGS, log_name)
        if not os.path.exists(log_path):
            continue
        method = log_name.replace("baseline_", "").replace(".log", "").replace("fuzz_", "GAN ")
        method = method.replace("mutation", "Mutation").replace("random", "Random")
        if "GAN" in method:
            method = "GAN_Retrained"

        entries = parse_log(log_path)
        # Find exception-triggering cases
        exception_entries = []
        for e in entries:
            if e["status"] == "SUCCESS" and e["response"]:
                fc, exc = parse_modbus_response(e["response"])
                if fc is not None and fc >= 0x80:
                    exception_entries.append(e)

        report.append(f"\n--- {method} Exception Case Studies ---")
        selected = exception_entries[:8]
        report.append(f"\n  ({len(exception_entries)} exceptions total, showing {len(selected)} representative cases):")
        for i, e in enumerate(selected):
            raw_sent = bytes.fromhex(e["hex_msg"])
            raw_resp = bytes.fromhex(e["response"])
            fc_sent = raw_sent[7] if len(raw_sent) > 7 else 0
            fc_resp = raw_resp[7] if len(raw_resp) > 7 else 0
            exc_code = raw_resp[8] if len(raw_resp) > 8 and fc_resp >= 0x80 else 0
            exc_name = MODBUS_EXCEPTION_NAMES.get(exc_code, f"Unknown")
            orig_func_name = MODBUS_FUNCTION_NAMES.get(fc_sent, f"0x{fc_sent:02X}")

            report.append(f"\n  Case {i+1}:")
            report.append(f"    TX: {e['hex_msg'][:60]}... ({len(raw_sent)} bytes)")
            report.append(f"    RX: {e['response'][:40]}... ({len(raw_resp)} bytes)")
            report.append(f"    Sent Function Code: 0x{fc_sent:02X} ({orig_func_name})")
            report.append(f"    Response Function Code: 0x{fc_resp:02X}")
            report.append(f"    Exception Code: 0x{exc_code:02X} ({exc_name})")

            # Annotate key bytes in the sent message
            if len(raw_sent) >= 8:
                tid = raw_sent[0:2].hex()
                pid = raw_sent[2:4].hex()
                leng = raw_sent[4:6].hex()
                uid = f"0x{raw_sent[6]:02X}"
                fc = f"0x{raw_sent[7]:02X}"
                report.append(f"    MBAP: [TID={tid}, PID={pid}, Len={leng}, UID={uid}]")
                report.append(f"    PDU: [FC={fc}, Data={raw_sent[8:12].hex() if len(raw_sent) > 8 else '(empty)'}]")

    report.append("\n\n" + "=" * 70)
    report.append("END OF ANALYSIS")
    report.append("=" * 70)

    output = '\n'.join(report)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)


if __name__ == "__main__":
    analyze()