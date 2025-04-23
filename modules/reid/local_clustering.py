import os
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def estimate_eps(features, min_samples=2, method='percentile', value=70):
    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(features)
    distances, _ = neigh.kneighbors(features)
    k_distances = distances[:, -1]
    if method == 'mean':
        return np.mean(k_distances)
    elif method == 'median':
        return np.median(k_distances)
    elif method == 'percentile':
        return np.percentile(k_distances, value)
    else:
        raise ValueError("method must be one of ['mean', 'median', 'percentile']")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_base = os.path.join(base_dir, "output")

zones = ['z1', 'z2']

for zone in zones:
    print(f"\n=== Processing {zone.upper()} ===")

    keypoint_dir = os.path.join(output_base, f"bbox_keypoint_npy_{zone}")
    original_features = np.load(os.path.join(output_base, f"zone{zone[-1]}_features.npy"))
    names = np.load(os.path.join(output_base, f"zone{zone[-1]}_feature_names.npy"))

    pca = PCA(n_components=5)
    features = pca.fit_transform(original_features)
    print(f"PCA → shape: {features.shape}, explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    cos_matrix = cosine_similarity(features)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cos_matrix, xticklabels=False, yticklabels=False, cmap="coolwarm", cbar=True)
    plt.title(f"{zone.upper()} Cosine Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_base, f"{zone}_cosine_similarity_heatmap.png"))
    plt.close()

    euclidean_matrix_full = np.linalg.norm(features[:, None] - features[None, :], axis=2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(euclidean_matrix_full, xticklabels=False, yticklabels=False, cmap="viridis", cbar=True)
    plt.title(f"{zone.upper()} Euclidean Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_base, f"{zone}_euclidean_distance_heatmap.png"))
    plt.close()

    min_samples = 10
    eps = 0.6
    print(f"[{zone}] eps: {eps:.4f}")
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(features)
    local_labels = clustering.labels_

    unique_labels = sorted(set(local_labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)

    if len(unique_labels) >= 2:
        local_cluster_means = [np.mean(features[local_labels == lbl], axis=0) for lbl in unique_labels]
        local_cluster_means = np.stack(local_cluster_means)
        similarity_matrix = cosine_similarity(local_cluster_means)
        euclidean_matrix = np.linalg.norm(local_cluster_means[:, None] - local_cluster_means[None, :], axis=2)

        cosine_threshold = np.percentile(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)], 95)
        l2_threshold = np.percentile(euclidean_matrix[np.triu_indices_from(euclidean_matrix, k=1)], 5)

        parent = {lbl: lbl for lbl in unique_labels}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px

        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                if similarity_matrix[i, j] >= cosine_threshold or euclidean_matrix[i, j] <= l2_threshold:
                    union(unique_labels[i], unique_labels[j])

        merged_ids = {}
        new_label = 0
        for lbl in unique_labels:
            root = find(lbl)
            if root not in merged_ids:
                merged_ids[root] = new_label
                new_label += 1

        local_final_labels = np.array([merged_ids[find(lbl)] if lbl != -1 else -1 for lbl in local_labels])
    else:
        print(f"[{zone}] 클러스터 수 부족, 병합 생략")
        local_final_labels = local_labels.copy()

    output_json_path = os.path.join(output_base, f"{zone}_local_clustering_result.json")
    json_data = []

    for i, name in enumerate(names):
        if local_final_labels[i] == -1:
            continue

        name_str = name.decode() if isinstance(name, bytes) else name
        try:
            parts = name_str.replace(".jpg", "").split("_")
            cam = int(parts[1][1:])   
            frame = int(parts[2][1:]) 
        except (IndexError, ValueError):
            print(f"이름 파싱 실패 → {name_str}")
            continue

        bbox_path = os.path.join(keypoint_dir, name_str.replace(".jpg", ".npy"))
        if not os.path.exists(bbox_path):
            print(f"bbox 파일 없음 → {bbox_path}")
            continue
        bbox_data = np.load(bbox_path, allow_pickle=True).item()
        bbox = bbox_data.get("bbox", [-1, -1, -1, -1])

        json_data.append({
            "zone": int(zone[-1]),
            "cam": cam,
            "frame": frame,
            "local_id": int(local_final_labels[i]),
            "bbox": list(map(int, bbox)),
            "feature": original_features[i].tolist()
        })

    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON 저장 완료 → {output_json_path}")

    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(12, 8))
    unique_final_labels = set(local_final_labels)
    colors = cm.tab10(np.linspace(0, 1, len(unique_final_labels)))

    for label, color in zip(unique_final_labels, colors):
        idx = (local_final_labels == label)
        label_name = f"{zone.upper()} ID {label}" if label != -1 else "Noise"
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], s=60, c=[color], label=label_name, alpha=0.7, edgecolors='k')
        for j in np.where(idx)[0]:
            plt.text(features_2d[j, 0]+0.5, features_2d[j, 1]+0.5, str(names[j]), fontsize=6)

    plt.title(f"{zone.upper()} Result (Local ID total num: {len(set(local_final_labels)) - (1 if -1 in local_final_labels else 0)})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    tsne_path = os.path.join(output_base, f"{zone}_id_clusters.png")
    plt.savefig(tsne_path)
    plt.close()
    print(f"t-SNE 저장 완료 → {tsne_path}")
