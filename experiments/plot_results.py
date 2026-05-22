# experiments/plot_results.py
import os
import sys
import json
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import REPORT_DIR, FIGURE_DIR

def plot_comparison(report_file):
    with open(report_file, 'r') as f:
        data = json.load(f)
    methods = list(data.keys())
    tirr = [data[m]["TIRR"] for m in methods]
    aovd = [data[m]["AoVD"] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    ax1.bar(methods, tirr, color=['gray','orange','blue'])
    ax1.set_ylabel("TIRR (%)")
    ax1.set_title("Test Input Rejection Rate (lower better)")
    for i, v in enumerate(tirr):
        ax1.text(i, v+1, f"{v:.1f}%", ha='center')
    ax2.bar(methods, aovd, color=['gray','orange','blue'])
    ax2.set_ylabel("AoVD (per 100 cases)")
    ax2.set_title("Anomaly Detection Ability (higher better)")
    for i, v in enumerate(aovd):
        ax2.text(i, v+0.2, f"{v:.2f}", ha='center')
    plt.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    save_path = os.path.join(FIGURE_DIR, "comparison.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    report_file = os.path.join(REPORT_DIR, "comparison.json")
    if not os.path.exists(report_file):
        print(f"Report not found: {report_file}. Please run experiments/run_compare.py first.")
    else:
        plot_comparison(report_file)