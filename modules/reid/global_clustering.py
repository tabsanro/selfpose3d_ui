import os
import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, "output")

zone1_path = os.path.join(output_dir, "z1_local_clustering_result.json")
zone2_path = os.path.join(output_dir, "z2_local_clustering_result.json")
output_json = os.path.join(output_dir, "global_clustering_result.json")
tsne_path = os.path.join(output_dir, "global_clustering_tsne_with_labels.png")

with open(zone1_path, "r") as f:
    data_z1 = json.load(f)
with open(zone2_path, "r") as f:
    data_z2 = json.load(f)

all_data = data_z1 + data_z2
features = np.array([item["feature"] for item in all_data])
print(f"총 feature 개수: {features.shape[0]}")

eps = 0.03
min_samples = 5
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
labels = dbscan.fit_predict(features)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Global Clustering 결과 - 클러스터 수: {n_clusters}, 노이즈 수: {n_noise}")

output_data = []
for item, global_id in zip(all_data, labels):
    if global_id == -1:
        continue
    output_data.append({
        "global_id": int(global_id),
        "zone": int(item["zone"]),
        "cam": int(item["cam"]),
        "frame": int(item["frame"]),
        "bbox": item["bbox"],
        # "feature": item["feature"]
    })

with open(output_json, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"Global Clustering 결과 저장 완료 → {output_json}")

tsne = TSNE(n_components=2, perplexity=10, random_state=42)
features_2d = tsne.fit_transform(features)

plt.figure(figsize=(12, 8))
unique_labels = sorted(set(labels))
cmap = plt.cm.get_cmap("tab10", len(unique_labels))

for label in unique_labels:
    idx = np.where(labels == label)[0]
    color = 'gray' if label == -1 else cmap(label)
    label_name = "Noise" if label == -1 else f"ID {label}"
    plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=label_name, alpha=0.6, s=60, c=[color])

    if label != -1:
        for i in idx:
            item = all_data[i]
            zone = item.get("zone", 0)
            cam = item["cam"]
            frame = item["frame"]
            label_text = f"z{zone}_c{cam}_f{frame}_g{label}"
            plt.text(features_2d[i, 0]+0.3, features_2d[i, 1]+0.3, label_text, fontsize=6)

plt.title(f"Global Clustering (Total IDs: {n_clusters})")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(tsne_path)
plt.show()
print(f"t-SNE 저장 완료 → {tsne_path}")
