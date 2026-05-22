import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.train_cluster import train_single_cluster

CLUSTER_ROOT = "data/clusters"
OUTPUT_MODEL_DIR = "model/checkpoints_improved"

ALL_CLUSTER_TYPES = ["no_clustering", "same_length", "advanced"]


def train_clusters(cluster_types, cluster_root=CLUSTER_ROOT, output_dir=OUTPUT_MODEL_DIR):
    for cluster_name in cluster_types:
        cluster_dir = os.path.join(cluster_root, cluster_name)
        if not os.path.exists(cluster_dir):
            print(f"Skip {cluster_name}: directory not found at {cluster_dir}")
            continue
        if cluster_name == "no_clustering":
            if os.path.exists(os.path.join(cluster_dir, "data.npy")):
                train_single_cluster(cluster_dir, output_dir, cluster_name, "0")
            else:
                subdir = os.path.join(cluster_dir, "class_0")
                if os.path.exists(subdir):
                    train_single_cluster(subdir, output_dir, cluster_name, "0")
        else:
            for subdir in os.listdir(cluster_dir):
                if subdir.startswith("class_"):
                    class_idx = subdir.split("_")[-1]
                    full_path = os.path.join(cluster_dir, subdir)
                    if os.path.exists(os.path.join(full_path, "data.npy")):
                        train_single_cluster(full_path, output_dir, cluster_name, class_idx)


def main():
    parser = argparse.ArgumentParser(description='Multi-cluster GAN training')
    parser.add_argument('--cluster_type', type=str, default='all',
                        choices=['all', 'no_clustering', 'same_length', 'advanced'],
                        help='Cluster type to train (default: all)')
    parser.add_argument('--cluster_root', type=str, default=CLUSTER_ROOT,
                        help='Root directory of cluster data')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_MODEL_DIR,
                        help='Output directory for model checkpoints')
    args = parser.parse_args()

    if args.cluster_type == 'all':
        types = ALL_CLUSTER_TYPES
    else:
        types = [args.cluster_type]

    train_clusters(types, args.cluster_root, args.output_dir)
    print("Training Finished.")


if __name__ == "__main__":
    main()