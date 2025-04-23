import os
import sys
import numpy as np
import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "fast-reid"))

from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data.transforms import build_transforms

def procrustes_distance(A, B):
    try:
        A_mean, B_mean = A.mean(axis=0), B.mean(axis=0)
        A_centered, B_centered = A - A_mean, B - B_mean
        norm_A, norm_B = np.linalg.norm(A_centered), np.linalg.norm(B_centered)
        A_scaled, B_scaled = A_centered / norm_A, B_centered / norm_B
        U, _, Vt = np.linalg.svd(B_scaled.T @ A_scaled)
        R = U @ Vt
        A_aligned = A_scaled @ R.T
        return np.sqrt(((A_aligned - B_scaled) ** 2).sum())
    except Exception:
        return np.nan

def load_keypoints(image_files, keypoint_dir):
    keypoints_list, valid_files = [], []
    for img_file in image_files:
        key_path = os.path.join(keypoint_dir, img_file.replace(".jpg", ".npy"))
        if os.path.exists(key_path):
            data = np.load(key_path, allow_pickle=True).item()
            keypoints = data["keypoints"]
            keypoints_list.append(keypoints)
            valid_files.append(img_file)
        else:
            print(f"[WARNING] Keypoint 파일 없음: {key_path}")
    return valid_files, keypoints_list

def build_fastreid_model(cfg_path, weights_path):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(cfg)
    model.eval()
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    transform = build_transforms(cfg, is_train=False)
    return model, transform

@torch.no_grad()
def extract_feature(model, transform, img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)
    outputs = model(img_tensor)
    return outputs.cpu().numpy().squeeze(0)

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_base = os.path.join(base_dir, "output")

cfg_path = os.path.join(base_dir, "fast-reid", "configs", "Market1501", "bagtricks_R50-ibn.yml")
weights_path = os.path.join(base_dir, "fast-reid", "weights", "market_bot_R50-ibn.pth")
model, transform = build_fastreid_model(cfg_path, weights_path)

datasets = {
    "1": {"img_dir": "cropped_image_z1", "key_dir": "bbox_keypoint_npy_z1"},
    "2": {"img_dir": "cropped_image_z2", "key_dir": "bbox_keypoint_npy_z2"}
}

alpha = 1.0

for zone_id, paths in datasets.items():
    cropped_image_dir = os.path.join(output_base, paths["img_dir"])
    keypoint_dir = os.path.join(output_base, paths["key_dir"])

    output_feature_path = os.path.join(output_base, f"zone{zone_id}_features.npy")
    output_name_path = os.path.join(output_base, f"zone{zone_id}_feature_names.npy")
    output_weight_path = os.path.join(output_base, f"zone{zone_id}_procrustes_weights.npy")
    output_plot_path = os.path.join(output_base, f"zone{zone_id}_procrustes_weight_histogram.png")

    image_files = sorted([f for f in os.listdir(cropped_image_dir) if f.endswith('.jpg')])
    valid_files, all_keypoints = load_keypoints(image_files, keypoint_dir)

    all_features = []
    procrustes_weights = []

    print(f"\n[Zone {zone_id}] {len(valid_files)} valid images")

    for i, img_file in enumerate(tqdm(valid_files, desc=f"Extracting features (zone{zone_id})")):
        img_path = os.path.join(cropped_image_dir, img_file)
        feat = extract_feature(model, transform, img_path)

        ref_kp = all_keypoints[i]
        dists = [
            procrustes_distance(ref_kp, other_kp)
            for j, other_kp in enumerate(all_keypoints) if j != i
        ]
        dists = [d for d in dists if not np.isnan(d)]
        weight = 1 + alpha * np.mean(dists) if dists else 1.0

        all_features.append(feat * weight)
        procrustes_weights.append(weight)

        print(f"Extracted: {img_file} | Weight = {weight:.3f}")

    np.save(output_feature_path, np.stack(all_features))
    np.save(output_name_path, np.array(valid_files))
    np.save(output_weight_path, np.array(procrustes_weights))

    print(f"Feature 저장 완료: {output_feature_path}")
    print(f"이름 저장 완료: {output_name_path}")
    print(f"Weight 저장 완료: {output_weight_path}")

    plt.figure(figsize=(8, 5))
    plt.hist(procrustes_weights, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of Procrustes-based Weights (Zone {zone_id})")
    plt.xlabel("Weight")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Histogram 저장 완료: {output_plot_path}")

print("\n모든 Zone 처리 완료")
