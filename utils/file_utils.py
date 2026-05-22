# utils/file_utils.py
import os
import json

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_text(lines, path):
    with open(path, 'w') as f:
        for line in lines:
            f.write(str(line) + '\n')

def load_text(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)