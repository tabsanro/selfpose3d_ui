import cv2
import os
import numpy as np
from ultralytics import YOLO

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_base = os.path.join(base_dir, "output")

model = YOLO(os.path.join(base_dir, "yolo11n-pose.pt")).to('cuda')

datasets = {
    "z1": os.path.join(output_base, "wild_image_z1"),
    "z2": os.path.join(output_base, "wild_image_z2")
}

SELECTED_KEYPOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
POSE_CONNECTIONS = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12)
]

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2
    xi1, yi1 = max(x1, x1p), max(y1, y1p)
    xi2, yi2 = min(x2, x2p), min(y2, y2p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2p - x1p) * (y2p - y1p)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

for zone_suffix, image_dir in datasets.items():
    bbox_keypoint_dir = os.path.join(output_base, f"bbox_keypoint_npy_{zone_suffix}")
    cropped_image_dir = os.path.join(output_base, f"cropped_image_{zone_suffix}")
    keypoint_image_dir = os.path.join(output_base, f"keypoint_image_{zone_suffix}")
    os.makedirs(bbox_keypoint_dir, exist_ok=True)
    os.makedirs(cropped_image_dir, exist_ok=True)
    os.makedirs(keypoint_image_dir, exist_ok=True)

    image_files = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.endswith('.jpg')]

    for img_path in image_files:
        img_name = os.path.basename(img_path)

        name_parts = img_name.split('_')
        try:
            cam = int(name_parts[2])
            frame_num = int(name_parts[3])
        except (IndexError, ValueError):
            print(f"[WARNING] 파일명 파싱 실패: {img_name}")
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[WARNING] 이미지 로드 실패: {img_path}")
            continue

        original_frame = frame.copy()
        results = model(frame)
        all_bboxes = []
        person_idx_global = 0

        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()

            for person_idx, person_kps in enumerate(keypoints):
                if person_idx >= len(boxes):
                    continue

                valid_keypoints = [(i, kp) for i, kp in enumerate(person_kps)
                                   if i in SELECTED_KEYPOINTS and kp[0] > 0 and kp[1] > 0]
                if len(valid_keypoints) < 12:
                    continue

                pose_vec = []
                for i in SELECTED_KEYPOINTS:
                    x, y = person_kps[i]
                    pose_vec.append([x, y] if x > 0 and y > 0 else [-1, -1])
                pose_vec = np.array(pose_vec)

                x_min, y_min, x_max, y_max = map(int, boxes[person_idx])
                if any(compute_iou((x_min, y_min, x_max, y_max), prev) > 0.05 for prev in all_bboxes):
                    continue
                all_bboxes.append((x_min, y_min, x_max, y_max))

                save_name = f"{zone_suffix}_c{cam}_f{frame_num}_p{person_idx_global}.jpg"
                cropped_path = os.path.join(cropped_image_dir, save_name)
                keypoint_path = os.path.join(bbox_keypoint_dir, save_name.replace(".jpg", ".npy"))
                vis_path = os.path.join(keypoint_image_dir, save_name)

                cv2.imwrite(cropped_path, original_frame[y_min:y_max, x_min:x_max])
                np.save(keypoint_path, {
                    'keypoints': pose_vec,
                    'bbox': [x_min, y_min, x_max, y_max]
                })

                for pt in pose_vec:
                    if pt[0] > 0 and pt[1] > 0:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                for a, b in POSE_CONNECTIONS:
                    if person_kps[a][0] > 0 and person_kps[b][0] > 0:
                        pt1 = tuple(map(int, person_kps[a]))
                        pt2 = tuple(map(int, person_kps[b]))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(frame, f"Person {person_idx_global}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                person_idx_global += 1

        vis_frame_path = os.path.join(keypoint_image_dir, f"{zone_suffix}_c{cam}_f{frame_num}.jpg")
        cv2.imwrite(vis_frame_path, frame)
        print(f"[{zone_suffix}] 저장 완료 → {vis_frame_path}")

print("\nYOLO Pose 추출 완료")
