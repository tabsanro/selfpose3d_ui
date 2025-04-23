import cv2
import os
import time

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_base = os.path.join(base_dir, "output")

video_dirs = {
    "wild_image_z1": os.path.join(base_dir, "data_0417", "hdVideos"),
    "wild_image_z2": os.path.join(base_dir, "data_0417_2", "hdVideos")
}

video_files = [
    "hd_00_00.mp4",
    "hd_00_01.mp4",
    "hd_00_02.mp4",
    "hd_00_03.mp4"
]

frame_interval = 1
total_images_per_video = 100

for subfolder, video_dir in video_dirs.items():
    save_dir = os.path.join(output_base, subfolder)
    os.makedirs(save_dir, exist_ok=True)

    deleted = 0
    for f in os.listdir(save_dir):
        if f.endswith(".jpg"):
            os.remove(os.path.join(save_dir, f))
            deleted += 1
    print(f"[INFO] '{subfolder}' 기존 이미지 {deleted}개 삭제 완료")

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[ERROR] 비디오를 열 수 없습니다: {video_path}")
            continue

        print(f"[INFO] 처리 중: {video_file} in {video_dir}")
        frame_count = 0
        saved_count = 0
        cam_id = os.path.splitext(video_file)[0]

        while saved_count < total_images_per_video:
            ret, frame = cap.read()
            if not ret:
                print(f"[WARNING] 프레임을 읽을 수 없습니다: {video_file}")
                break

            if frame_count % frame_interval == 0:
                timestamp = int(time.time() * 1000)
                filename = f"{cam_id}_{frame_count}_{timestamp}.jpg"
                image_path = os.path.join(save_dir, filename)
                cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                print(f"    Saved: {image_path}")
                saved_count += 1

            frame_count += 1

        cap.release()

cv2.destroyAllWindows()
print("\n[완료] 모든 이미지 저장 완료")
