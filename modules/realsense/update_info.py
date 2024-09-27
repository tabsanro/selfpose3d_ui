# 1. 리얼센스 카메라 감지
# 2. 각 카메라의 시리얼 번호를 가져와 파이프라인 설정
# 3. 한 대의 카메라 영상을 가져와서 캘리브레이션을 진행함
# --| c 버튼을 누르면 이미지가 저장되고 q 버튼을 누르면 내부 파라미터를 구함
# 4. json 형식으로 시리얼 넘버와 내부 파라미터를 저장함
# 5. 모든 연결된 카메라에 대해 반복함

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
import os.path as osp
import glob

base_dir = osp.dirname(osp.abspath(__file__))
img_dir = osp.join(base_dir, 'captures')

CHECKERBOARD = (7, 9)

def list_cameras():
    """Detects all connected RealSense cameras and returns their serial numbers."""
    context = rs.context()
    devices = context.query_devices()
    serial_numbers = [device.get_info(rs.camera_info.serial_number) for device in devices]
    return serial_numbers

def configure_camera(serial_number):
    """Configures a RealSense camera pipeline for the given serial number."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def calibrate_camera(pipeline, serial_number):
    """Captures frames from the camera and performs calibration."""
    print(f"Press 'c' to capture an image, 'q' to calculate internal parameters for camera {serial_number}.")
    i = 0
    dir = osp.join(img_dir, serial_number)
    os.makedirs(dir, exist_ok=True)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense Calibration', color_image)
            
            key = cv2.waitKey(1)
            if key == ord('c'):
                # cv2.imwrite('calibration_image.png', color_image)
                cv2.imwrite(osp.join(dir, f'{i:02d}.png'), color_image)
                i += 1
                print(f'#{i} image saved.')
            elif key == ord('q'):
                print('Calculating internal parameters...')
                image_files = glob.glob(osp.join(dir, '*.png'))
                if not image_files:
                    print('No images captured. Please try again.')
                    continue

                objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

                # 각 포인트의 위치를 실제 크기에 맞게 설정
                objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

                # 3D와 2D 포인트를 저장할 배열
                objpoints = []  # 3D 포인트
                imgpoints = []  # 2D 포인트

                for image_file in image_files:
                    if not os.path.exists(image_file):
                        print(f"File does not exist: {image_file}")
                        continue

                    img = cv2.imread(image_file)
                    if img is None:
                        print(f"Failed to load image: {image_file}")
                        continue

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # 체스보드 코너 찾기
                    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
                    
                    if ret:
                        objpoints.append(objp)
                        imgpoints.append(corners)

                print(f"Number of valid images with detected corners: {len(objpoints)}")

                # 결과 출력
                if len(objpoints) > 0 and len(imgpoints) > 0:
                    # 카메라 캘리브레이션 수행
                    ret, camera_matrix, dist_coeffs, _, _= cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                    print("ret:", ret)
                    print("Camera matrix:")
                    print(camera_matrix)
                    print("\nDistortion coefficients:")
                    print(dist_coeffs)
                    internal_params = {
                        'camera_matrix': camera_matrix.tolist(),
                        'dist_coeffs': dist_coeffs.tolist()
                    }

                return internal_params
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def save_calibration(serial_number, internal_params):
    """Saves the camera serial number and internal parameters to a JSON file, appending if it exists."""
    # 기존 파일이 있는지 확인하고 데이터를 로드
    file_path = osp.join(base_dir, 'realsense_info.json')
    if osp.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    # 같은 serial_number가 있는지 확인
    existing_entry = next((item for item in data if item['serial_number'] == serial_number), None)
    
    if existing_entry:
        # 사용자에게 덮어쓸지 여부를 물어봄
        overwrite = input(f'Serial number {serial_number} already exists. Do you want to overwrite it? (y/n): ').strip().lower()
        if overwrite != 'y':
            print('Calibration data not saved.')
            return

        # 기존 항목 업데이트
        existing_entry['internal_parameters'] = internal_params
    else:
        # 새로운 데이터를 추가
        data.append({
            'serial_number': serial_number,
            'internal_parameters': internal_params
        })

    # 파일에 데이터를 저장 (덮어쓰기)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'Calibration data saved for camera {serial_number}.')

def main():
    serial_numbers = list_cameras()
    if not serial_numbers:
        print("No RealSense cameras detected.")
        return
    
    for serial_number in serial_numbers:
        print(f"Configuring camera {serial_number}...")
        pipeline = configure_camera(serial_number)
        
        print("Starting calibration...")
        internal_params = calibrate_camera(pipeline, serial_number)
        
        print("Saving calibration data...")
        save_calibration(serial_number, internal_params)
    
    print("Calibration process completed for all cameras.")

if __name__ == "__main__":
    main()
