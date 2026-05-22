import os
import sys
import json
import threading
import subprocess
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLUSTER_DIR = os.path.join(DATA_DIR, "clusters")
MODEL_DIR = os.path.join(BASE_DIR, "model", "checkpoints_improved")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
VALID_OUTPUT_DIR = os.path.join(BASE_DIR, "valid_output")
RETRAINED_OUTPUT_DIR = os.path.join(BASE_DIR, "valid_output_retrained")
PCAP_DIR = os.path.join(BASE_DIR, "pcap")

training_process = None
training_log = []
fuzzing_process = None
fuzzing_log = []
iterative_process = None
iterative_log = []
iterative_status = {"running": False, "current_iteration": 0, "total_iterations": 0, "history": []}


def safe_read_lines(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return [line.strip() for line in f if line.strip()]


def safe_read_json(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def safe_read_text(filepath):
    if not os.path.exists(filepath):
        return ""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def get_cluster_info():
    import numpy as np
    info = {}
    for cluster_type in ["no_clustering", "same_length", "advanced"]:
        cluster_path = os.path.join(CLUSTER_DIR, cluster_type)
        if not os.path.exists(cluster_path):
            continue
        if cluster_type == "no_clustering":
            data_file = os.path.join(cluster_path, "data.npy")
            if os.path.exists(data_file):
                data = np.load(data_file)
                info[cluster_type] = {
                    "classes": [{"id": 0, "samples": int(data.shape[0]), "max_len": int(data.shape[1])}]
                }
        else:
            classes = []
            for item in sorted(os.listdir(cluster_path)):
                if item.startswith("class_"):
                    class_path = os.path.join(cluster_path, item)
                    data_file = os.path.join(class_path, "data.npy")
                    if os.path.exists(data_file):
                        data = np.load(data_file)
                        classes.append({
                            "id": int(item.split("_")[1]),
                            "samples": int(data.shape[0]),
                            "max_len": int(data.shape[1])
                        })
            info[cluster_type] = {"classes": classes}
    return info


def get_model_info():
    info = {}
    for cluster_type in ["no_clustering", "same_length", "advanced"]:
        model_path = os.path.join(MODEL_DIR, cluster_type)
        if not os.path.exists(model_path):
            continue
        classes = []
        for item in sorted(os.listdir(model_path)):
            if item.startswith("class_"):
                class_path = os.path.join(model_path, item)
                if not os.path.isdir(class_path):
                    continue
                checkpoints = [f for f in os.listdir(class_path) if f.endswith(".pth")]
                has_retrain = os.path.exists(os.path.join(class_path, "retrain", "generator_retrained.pth"))
                classes.append({
                    "id": int(item.split("_")[1]),
                    "checkpoints": sorted(checkpoints),
                    "has_retrained": has_retrain
                })
        info[cluster_type] = {"classes": classes}
    return info


def scan_test_case_files(directories):
    files = []
    for dir_path in directories:
        if not os.path.exists(dir_path):
            continue
        dir_name = os.path.basename(dir_path)
        for f in os.listdir(dir_path):
            if f.endswith('.txt'):
                path = os.path.join(dir_path, f)
                lines = safe_read_lines(path)
                files.append({
                    "name": f,
                    "dir": dir_name,
                    "path": path,
                    "count": len(lines),
                    "preview": lines[:5],
                    "mtime": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
                })
    return sorted(files, key=lambda x: x["mtime"], reverse=True)


def scan_log_files(directories):
    logs = []
    for dir_path in directories:
        if not os.path.exists(dir_path):
            continue
        dir_name = os.path.basename(dir_path)
        if not os.path.exists(dir_path):
            continue
        for f in os.listdir(dir_path):
            if f.endswith('.log'):
                path = os.path.join(dir_path, f)
                content = safe_read_text(path)
                line_count = content.count('\n') + 1
                logs.append({
                    "name": f,
                    "dir": dir_name,
                    "path": path,
                    "lines": line_count,
                    "size": os.path.getsize(path),
                    "preview": content[:2000],
                    "mtime": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
                })
    return sorted(logs, key=lambda x: x["mtime"], reverse=True)


def scan_report_files(directories):
    reports = []
    for dir_path in directories:
        if not os.path.exists(dir_path):
            continue
        dir_name = os.path.basename(dir_path)
        for f in os.listdir(dir_path):
            if f.endswith('.json'):
                path = os.path.join(dir_path, f)
                data = safe_read_json(path)
                reports.append({
                    "name": f,
                    "dir": dir_name,
                    "path": path,
                    "data": data,
                    "mtime": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
                })
    return sorted(reports, key=lambda x: x["mtime"], reverse=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    return jsonify({"status": "running", "timestamp": datetime.now().isoformat()})


@app.route('/api/data/overview')
def api_data_overview():
    raw_msgs = safe_read_lines(os.path.join(DATA_DIR, "raw_messages.txt"))
    merged_msgs = safe_read_lines(os.path.join(DATA_DIR, "merged_raw_messages.txt"))

    pcap_files = []
    if os.path.exists(PCAP_DIR):
        pcap_files = sorted([f for f in os.listdir(PCAP_DIR) if f.endswith(('.pcap', '.pcapng'))])

    cluster_info = get_cluster_info()
    model_info = get_model_info()

    total_clusters = sum(len(v["classes"]) for v in cluster_info.values())
    total_models = sum(len(v["classes"]) for v in model_info.values())

    anomaly_seeds = safe_read_lines(os.path.join(VALID_OUTPUT_DIR, "anomaly_seeds.txt"))

    all_reports = scan_report_files([
        os.path.join(OUTPUT_DIR, "reports"),
        os.path.join(VALID_OUTPUT_DIR, "reports"),
        os.path.join(RETRAINED_OUTPUT_DIR, "reports"),
    ])

    return jsonify({
        "raw_messages": len(raw_msgs),
        "merged_messages": len(merged_msgs),
        "pcap_files": len(pcap_files),
        "pcap_list": pcap_files,
        "clusters": cluster_info,
        "models": model_info,
        "total_clusters": total_clusters,
        "total_models": total_models,
        "anomaly_seeds": len(anomaly_seeds),
        "reports": all_reports
    })


@app.route('/api/models/list')
def api_models_list():
    return jsonify(get_model_info())


@app.route('/api/clusters/list')
def api_clusters_list():
    providers = []
    search_dirs = [
        ('data/clusters', '改进数据 (improved)'),
        ('data/clusters_original', '原始数据 (original)'),
    ]
    for rel_dir, label in search_dirs:
        abs_dir = os.path.join(BASE_DIR, rel_dir)
        if os.path.isdir(abs_dir):
            types = []
            for ct in ['no_clustering', 'same_length', 'advanced']:
                ct_dir = os.path.join(abs_dir, ct)
                if os.path.isdir(ct_dir):
                    classes = []
                    for sub in sorted(os.listdir(ct_dir)):
                        if sub.startswith('class_') and os.path.isfile(os.path.join(ct_dir, sub, 'data.npy')):
                            classes.append({'id': sub.split('_')[-1], 'path': os.path.join(rel_dir, ct, sub)})
                    if classes:
                        types.append({'name': ct, 'classes': classes, 'total': len(classes)})
            if types:
                providers.append({'path': rel_dir, 'label': label, 'types': types, 'total_types': len(types)})
    return jsonify({'providers': providers})


@app.route('/api/files/testcases')
def api_files_testcases():
    files = scan_test_case_files([
        os.path.join(OUTPUT_DIR, "test_cases"),
        os.path.join(VALID_OUTPUT_DIR, "test_cases"),
        os.path.join(VALID_OUTPUT_DIR, "valid_test_cases"),
        os.path.join(RETRAINED_OUTPUT_DIR, "test_cases"),
    ])
    return jsonify({"files": files, "total": len(files)})


@app.route('/api/files/logs')
def api_files_logs():
    logs = scan_log_files([
        os.path.join(OUTPUT_DIR, "logs"),
        os.path.join(VALID_OUTPUT_DIR, "logs"),
        os.path.join(RETRAINED_OUTPUT_DIR, "logs"),
    ])
    return jsonify({"logs": logs, "total": len(logs)})


@app.route('/api/files/reports')
def api_files_reports():
    reports = scan_report_files([
        os.path.join(OUTPUT_DIR, "reports"),
        os.path.join(VALID_OUTPUT_DIR, "reports"),
        os.path.join(RETRAINED_OUTPUT_DIR, "reports"),
    ])
    return jsonify({"reports": reports, "total": len(reports)})


@app.route('/api/file/content')
def api_file_content():
    filepath = request.args.get('path', '')
    if not os.path.exists(filepath):
        return jsonify({"error": "文件不存在"}), 404
    content = safe_read_text(filepath)
    lines = content.split('\n')
    return jsonify({
        "path": filepath,
        "name": os.path.basename(filepath),
        "total_lines": len(lines),
        "content": content[:50000],
        "preview": lines[:100]
    })


@app.route('/api/file/download')
def api_file_download():
    filepath = request.args.get('path', '')
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({"error": "文件不存在"}), 404


@app.route('/api/anomaly/seeds')
def api_anomaly_seeds():
    seeds = safe_read_lines(os.path.join(VALID_OUTPUT_DIR, "anomaly_seeds.txt"))
    return jsonify({"seeds": seeds, "count": len(seeds)})


@app.route('/api/evaluation/all')
def api_evaluation_all():
    results = {}
    for label, dir_path in [
        ("原始实验", os.path.join(OUTPUT_DIR, "reports", "comparison.json")),
        ("有效过滤", os.path.join(VALID_OUTPUT_DIR, "reports", "comparison.json")),
        ("反馈微调", os.path.join(RETRAINED_OUTPUT_DIR, "reports", "comparison.json")),
    ]:
        data = safe_read_json(dir_path)
        if data:
            results[label] = data
    return jsonify(results)


@app.route('/api/data/collect', methods=['POST'])
def api_data_collect():
    global training_log
    training_log = []

    data = request.json or {}
    use_pcap = data.get('use_pcap', True)
    synthetic_count = data.get('synthetic_count', 15000)
    output_file = data.get('output_file', os.path.join(DATA_DIR, "raw_messages.txt"))

    def run_collect():
        import subprocess
        script = os.path.join(BASE_DIR, "capture", "message_collector.py")
        proc = subprocess.Popen(
            [sys.executable, "-c", f"""
import sys; sys.path.insert(0, r'{BASE_DIR}')
from capture.message_collector import MessageCollector
from config import PCAP_PATHS, SYNTHETIC_MAX_LEN

if {use_pcap} and PCAP_PATHS:
    print("Extracting from pcap files...")
    msgs = MessageCollector.extract_from_multiple_pcaps(PCAP_PATHS)
else:
    msgs = []
    print("Skipping pcap extraction")

if not msgs or not {use_pcap}:
    print(f"Generating {synthetic_count} synthetic Modbus messages...")
    synth = MessageCollector.generate_synthetic_modbus({synthetic_count}, SYNTHETIC_MAX_LEN)
    msgs.extend(synth)
    msgs = list(dict.fromkeys(msgs))

from utils.file_utils import ensure_dir, save_text
ensure_dir(os.path.dirname(r'{output_file}'))
save_text(msgs, r'{output_file}')
print(f"Done. Saved {{len(msgs)}} messages to {output_file}")
"""],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in proc.stdout:
            training_log.append(line.strip())
        proc.wait()

    thread = threading.Thread(target=run_collect)
    thread.start()
    return jsonify({"status": "started", "message": f"数据采集已启动, 模拟生成{synthetic_count}条"})


@app.route('/api/data/pcap/list')
def api_data_pcap_list():
    pcap_files = []
    if os.path.exists(PCAP_DIR):
        for f in sorted(os.listdir(PCAP_DIR)):
            if f.endswith(('.pcap', '.pcapng')):
                path = os.path.join(PCAP_DIR, f)
                pcap_files.append({
                    "name": f,
                    "path": path,
                    "size": os.path.getsize(path)
                })
    return jsonify({"files": pcap_files, "count": len(pcap_files)})


@app.route('/api/data/quality/status')
def api_data_quality_status():
    improved_file = os.path.join(DATA_DIR, "improved_raw_messages.txt")
    improved_exists = os.path.exists(improved_file)
    improved_count = 0
    improved_fc_dist = {}
    improved_len_dist = {}
    if improved_exists:
        improved_msgs = safe_read_lines(improved_file)
        improved_count = len(improved_msgs)
        for msg in improved_msgs[:5000]:
            if len(msg) >= 14:
                try:
                    fc = int(msg[14:16], 16)
                    improved_fc_dist[str(fc)] = improved_fc_dist.get(str(fc), 0) + 1
                except:
                    pass
            blen = len(msg) // 2
            improved_len_dist[str(blen)] = improved_len_dist.get(str(blen), 0) + 1
    return jsonify({
        "improved_exists": improved_exists,
        "improved_count": improved_count,
        "improved_fc_dist": improved_fc_dist,
        "improved_len_dist": improved_len_dist
    })


@app.route('/api/data/quality/improve', methods=['POST'])
def api_data_quality_improve():
    global training_log
    training_log = []

    def run_improvement():
        script_path = os.path.join(BASE_DIR, "script", "improve_data_quality.py")
        process = subprocess.Popen(
            [sys.executable, script_path],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in process.stdout:
            training_log.append(line.strip())
        process.wait()

    thread = threading.Thread(target=run_improvement)
    thread.start()
    return jsonify({"status": "started", "message": "数据质量改进已启动"})


@app.route('/api/data/raw/info')
def api_data_raw_info():
    raw = safe_read_lines(os.path.join(DATA_DIR, "raw_messages.txt"))
    merged = safe_read_lines(os.path.join(DATA_DIR, "merged_raw_messages.txt"))
    improved_file = os.path.join(DATA_DIR, "improved_raw_messages.txt")
    improved = safe_read_lines(improved_file) if os.path.exists(improved_file) else []
    preview_raw = raw[:5] if raw else []
    preview_merged = merged[:5] if merged else []
    preview_improved = improved[:5] if improved else []

    func_dist = {}
    length_dist = {}
    for msg in raw[:5000]:
        if len(msg) >= 14:
            try:
                fc = int(msg[14:16], 16)
                func_dist[str(fc)] = func_dist.get(str(fc), 0) + 1
            except:
                pass
        blen = len(msg) // 2
        length_dist[str(blen)] = length_dist.get(str(blen), 0) + 1

    improved_fc_dist = {}
    improved_len_dist = {}
    for msg in improved[:5000]:
        if len(msg) >= 14:
            try:
                fc = int(msg[14:16], 16)
                improved_fc_dist[str(fc)] = improved_fc_dist.get(str(fc), 0) + 1
            except:
                pass
        blen = len(msg) // 2
        improved_len_dist[str(blen)] = improved_len_dist.get(str(blen), 0) + 1

    return jsonify({
        "raw_count": len(raw),
        "merged_count": len(merged),
        "improved_count": len(improved),
        "preview_raw": preview_raw,
        "preview_merged": preview_merged,
        "preview_improved": preview_improved,
        "func_dist_raw": func_dist,
        "length_dist_raw": length_dist,
        "func_dist_improved": improved_fc_dist,
        "length_dist_improved": improved_len_dist
    })


@app.route('/api/preprocessing/run', methods=['POST'])
def api_preprocessing_run():
    global training_log
    training_log = []

    def run_preprocessing():
        script_path = os.path.join(BASE_DIR, "script", "run_preprocessing.py")
        process = subprocess.Popen(
            [sys.executable, script_path],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in process.stdout:
            training_log.append(line.strip())
        process.wait()

    thread = threading.Thread(target=run_preprocessing)
    thread.start()
    return jsonify({"status": "started", "message": "预处理已启动"})


@app.route('/api/preprocessing/log')
def api_preprocessing_log():
    return jsonify({"log": training_log[-100:]})


@app.route('/api/train/start', methods=['POST'])
def api_train_start():
    global training_process, training_log
    training_log = []
    data = request.get_json(silent=True) or {}
    cluster_type = data.get('cluster_type', 'all')
    cluster_root = data.get('cluster_root', 'data/clusters')
    output_dir = data.get('output_dir', 'model/checkpoints_improved')

    def run_training():
        global training_process
        script_path = os.path.join(BASE_DIR, "train_all.py")
        args = [
            sys.executable, script_path,
            '--cluster_type', cluster_type,
            '--cluster_root', cluster_root,
            '--output_dir', output_dir,
        ]
        training_process = subprocess.Popen(
            args,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in training_process.stdout:
            training_log.append(line.strip())
        training_process.wait()
        training_process = None

    thread = threading.Thread(target=run_training)
    thread.start()
    return jsonify({"status": "started", "message": "训练已启动"})


@app.route('/api/train/stop', methods=['POST'])
def api_train_stop():
    global training_process
    if training_process:
        training_process.terminate()
        training_process = None
        return jsonify({"status": "stopped", "message": "训练已停止"})
    return jsonify({"status": "no_process", "message": "没有正在运行的训练进程"})


@app.route('/api/train/log')
def api_train_log():
    return jsonify({"log": training_log[-100:]})


@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    cluster_type = data.get('cluster_type', 'no_clustering')
    class_id = data.get('class_id', 0)
    num_cases = data.get('num_cases', 100)

    model_path = os.path.join(MODEL_DIR, cluster_type, f"class_{class_id}", "generator_step10.pth")
    if not os.path.exists(model_path):
        return jsonify({"error": "模型文件不存在，请先训练模型"}), 400

    data_path = os.path.join(CLUSTER_DIR, cluster_type)
    if cluster_type != "no_clustering":
        data_path = os.path.join(data_path, f"class_{class_id}")

    import torch
    import numpy as np
    from model.generator import Generator

    vocab = safe_read_json(os.path.join(data_path, "vocab.json"))
    vocab = {int(k) if k.isdigit() else k: v for k, v in vocab.items()}
    max_len = int(safe_read_text(os.path.join(data_path, "max_len.txt")).strip())
    pad_idx = vocab['<PAD>']
    vocab_size = max(vocab.values()) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = Generator(vocab_size, 256, 32, max_len, pad_idx)
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.to(device)

    start_token = 2
    samples = gen.sample(num_cases, start_token, device, max_len)

    inv_vocab = {v: k for k, v in vocab.items() if isinstance(k, int)}
    test_cases = []
    for seq in samples:
        byte_seq = []
        for idx in seq:
            if idx == pad_idx:
                break
            if idx in inv_vocab:
                byte_seq.append(inv_vocab[idx])
        if byte_seq:
            test_cases.append(bytes(byte_seq).hex())

    output_dir = os.path.join(OUTPUT_DIR, "test_cases")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"generated_{cluster_type}_class{class_id}.txt")
    with open(output_file, 'w') as f:
        for case in test_cases:
            f.write(case + '\n')

    return jsonify({
        "status": "success",
        "num_generated": len(test_cases),
        "output_file": output_file,
        "sample_cases": test_cases[:10]
    })


@app.route('/api/fuzzing/start', methods=['POST'])
def api_fuzzing_start():
    global fuzzing_log
    fuzzing_log = []

    data = request.json or {}
    test_file = data.get('test_file', '')
    target_ip = data.get('target_ip', '127.0.0.1')
    target_port = data.get('target_port', 502)

    if not os.path.exists(test_file):
        return jsonify({"error": "测试用例文件不存在"}), 400

    def run_fuzzing():
        from fuzzer.fuzzer_main import run_fuzzing as do_fuzzing
        log_file = os.path.join(OUTPUT_DIR, "logs", "fuzzing.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        do_fuzzing(test_file, log_file)
        with open(log_file, 'r') as f:
            for line in f:
                fuzzing_log.append(line.strip())

    thread = threading.Thread(target=run_fuzzing)
    thread.start()
    return jsonify({"status": "started", "message": "模糊测试已启动"})


@app.route('/api/fuzzing/log')
def api_fuzzing_log():
    return jsonify({"log": fuzzing_log[-100:]})


@app.route('/api/feedback/extract', methods=['POST'])
def api_feedback_extract():
    script_path = os.path.join(BASE_DIR, "feedback", "extract_anomalies_fixed.py")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR,
        capture_output=True,
        text=True
    )
    seeds = safe_read_lines(os.path.join(VALID_OUTPUT_DIR, "anomaly_seeds.txt"))
    return jsonify({
        "status": "success",
        "output": result.stdout,
        "error": result.stderr,
        "seed_count": len(seeds),
        "seeds": seeds[:20]
    })


@app.route('/api/feedback/retrain', methods=['POST'])
def api_feedback_retrain():
    global training_log
    training_log = []

    data = request.json or {}
    epochs = data.get('epochs', 3)

    script_path = os.path.join(BASE_DIR, "feedback", "mutate_and_retrain.py")

    def run_retrain():
        global training_process
        training_process = subprocess.Popen(
            [sys.executable, script_path],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in training_process.stdout:
            training_log.append(line.strip())
        training_process.wait()
        training_process = None

    thread = threading.Thread(target=run_retrain)
    thread.start()
    return jsonify({"status": "started", "message": f"反馈微调已启动 (epochs={epochs})"})


@app.route('/api/feedback/generate', methods=['POST'])
def api_feedback_generate():
    global training_log
    training_log = []

    retrained_model = os.path.join(
        MODEL_DIR, "no_clustering", "class_0", "retrain", "generator_retrained.pth"
    )
    if not os.path.exists(retrained_model):
        return jsonify({"error": "微调模型不存在，请先执行反馈微调"}), 400

    script_path = os.path.join(BASE_DIR, "model", "generate_from_retrained.py")

    def run_generate():
        global training_process
        training_process = subprocess.Popen(
            [sys.executable, script_path, "--model_path", retrained_model],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in training_process.stdout:
            training_log.append(line.strip())
        training_process.wait()
        training_process = None

    thread = threading.Thread(target=run_generate)
    thread.start()
    return jsonify({"status": "started", "message": "测试用例生成已启动"})


@app.route('/api/feedback/compare', methods=['POST'])
def api_feedback_compare():
    global training_log
    training_log = []

    script_path = os.path.join(BASE_DIR, "experiments", "run_retrained_compare.py")

    def run_compare():
        global training_process
        training_process = subprocess.Popen(
            [sys.executable, script_path],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in training_process.stdout:
            training_log.append(line.strip())
        training_process.wait()
        training_process = None

    thread = threading.Thread(target=run_compare)
    thread.start()
    return jsonify({"status": "started", "message": "对比评估已启动"})


@app.route('/api/feedback/iterative/start', methods=['POST'])
def api_feedback_iterative_start():
    global iterative_log, iterative_status
    iterative_log = []
    iterative_status = {"running": True, "current_iteration": 0, "total_iterations": 0, "history": []}

    data = request.json or {}
    num_iterations = data.get('iterations', 3)
    num_generate = data.get('num_generate', 2000)
    cluster_type = data.get('cluster_type', 'no_clustering')
    class_id = data.get('class_id', 0)
    retrain_epochs = data.get('retrain_epochs', 3)
    retrain_lr = data.get('retrain_lr', 1e-4)
    mutations_per_seed = data.get('mutations_per_seed', 8)
    target_ip = data.get('target_ip', '127.0.0.1')
    target_port = data.get('target_port', 502)

    iterative_status["total_iterations"] = num_iterations

    def run_iterative():
        global iterative_status
        from feedback.iterative_feedback import iterative_feedback_loop
        import builtins

        original_print = builtins.print

        def capturing_print(*args, **kwargs):
            text = ' '.join(str(a) for a in args)
            iterative_log.append(text)
            if '迭代第' in text and '/' in text:
                try:
                    import re
                    m = re.search(r'(\d+)/(\d+)', text)
                    if m:
                        iterative_status["current_iteration"] = int(m.group(1))
                except:
                    pass

        builtins.print = capturing_print
        try:
            result = iterative_feedback_loop(
                num_iterations=num_iterations,
                num_generate=num_generate,
                cluster_type=cluster_type,
                class_id=class_id,
                retrain_epochs=retrain_epochs,
                retrain_lr=retrain_lr,
                mutations_per_seed=mutations_per_seed,
                target_ip=target_ip,
                target_port=target_port
            )
            if result and result.get("history"):
                iterative_status["history"] = result["history"]
                iterative_status["final_model"] = result.get("final_model", "")
                iterative_status["final_test_cases"] = result.get("final_test_cases", "")
                iterative_status["run_dir"] = result.get("run_dir", "")
        except Exception as e:
            iterative_log.append(f"ERROR: {str(e)}")
        finally:
            builtins.print = original_print
            iterative_status["running"] = False

    thread = threading.Thread(target=run_iterative)
    thread.start()
    return jsonify({"status": "started", "message": f"迭代反馈调节已启动 (iterations={num_iterations})"})


@app.route('/api/feedback/iterative/status')
def api_feedback_iterative_status():
    return jsonify(iterative_status)


@app.route('/api/feedback/iterative/log')
def api_feedback_iterative_log():
    return jsonify({"log": iterative_log[-200:]})


@app.route('/api/feedback/iterative/history')
def api_feedback_iterative_history():
    feedback_dir = os.path.join(BASE_DIR, "iterative_feedback_output")
    runs = []
    if os.path.exists(feedback_dir):
        for run_name in sorted(os.listdir(feedback_dir), reverse=True):
            run_path = os.path.join(feedback_dir, run_name)
            history_file = os.path.join(run_path, "iteration_history.json")
            if os.path.exists(history_file):
                history = safe_read_json(history_file)
                runs.append({
                    "name": run_name,
                    "path": run_path,
                    "history": history,
                    "has_final_model": os.path.exists(os.path.join(run_path, "generator_final.pth"))
                })
    return jsonify({"runs": runs})


if __name__ == '__main__':
    for d in [OUTPUT_DIR, VALID_OUTPUT_DIR, RETRAINED_OUTPUT_DIR]:
        for sub in ["test_cases", "logs", "reports"]:
            os.makedirs(os.path.join(d, sub), exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
