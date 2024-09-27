# # # !pip install pyrealsense2
import sys
import pyrealsense2 as rs
import numpy as np
import json
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import cv2
import subprocess

class CameraWidget(QWidget):
    def __init__(self, path=None):
        super().__init__()
        self.move(0, 0)
        self.path = path
        self.camera_dict = self.open_json(self.path)
        self.pipelines = self.set_piplines(self.camera_dict)
        
        self.initUI()
        # QTimer 설정: 주기적으로 영상을 갱신합니다.
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)  # 30ms마다 갱신

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.marker_length = 1.39

    def initUI(self):
        # 레이아웃 설정
        self.layout = QGridLayout()
        self.labels = []
        # 카메라의 수에 따라 QLabel 생성
        num_cameras = len(self.pipelines)
        self.cols = 2  # 기본적으로 두 열로 설정
        self.rows = (num_cameras + self.cols - 1) // self.cols

        self.calibrate_button = QPushButton("Camera Calibration", self)
        self.calibrate_button.clicked.connect(lambda: self.save_json(self.path))
        self.layout.addWidget(self.calibrate_button, self.rows, 1)

        for i in range(num_cameras):
            label = QLabel(self)
            label.setScaledContents(True)
            self.labels.append(label)
            self.layout.addWidget(label, i // self.cols, i % self.cols)
        self.setLayout(self.layout)

    def open_json(self, path):
        # 카메라 내부 파라미터를 가져오기 위한 json 파일 열기
        with open(path) as f:
            return json.load(f)
    
    def save_json(self, path):
        # 켈리브레이션 정보 저장

        # self.pipelines 의 serial_number와 같은 키가 self.camera_dict에 있는지 확인
        # 있으면 self.camera_dict의 rvec, tvec을 업데이트
        for _, params in self.pipelines.items():
            serial_number = params['serial_number']
            matching_camera = next((camera for camera in self.camera_dict if camera['serial_number'] == serial_number), None)

            if matching_camera is not None:
                matching_camera['rvec'] = params['rvec']
                matching_camera['tvec'] = params['tvec']

        with open(path, 'w') as f:
            json.dump(self.camera_dict, f, indent=4)
        print("Calibration data saved")

  
    def set_piplines(self, camera_dict):
        # 리얼센스 파이프라인 설정
        def check_serial_number(serial_number):
            for camera in camera_dict:
                if camera['serial_number'] == serial_number:
                    return camera
            print("No calibration data found for camera", serial_number)
            return False
            
        pipelines = {}
        context = rs.context()
        devices = context.query_devices()
        print("Found", len(devices), "cameras")
        for device in devices:
            serial_number = device.get_info(rs.camera_info.serial_number)
            if not (camera := check_serial_number(serial_number)):
                continue
            pipeline = rs.pipeline()
            pipelines[pipeline] = {
                'serial_number': serial_number,
                'camera_matrix': np.array(camera['internal_parameters']['camera_matrix']),
                'dist_coeffs': np.array(camera['internal_parameters']['dist_coeffs']),
                'rvec': [],
                'tvec': []
            }
            config = rs.config()
            config.enable_device(serial_number)
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
            pipeline.start(config)

        print("Connected to", len(pipelines), "cameras")
        return pipelines

    def run_poc(self):
        # Function to run poc.py
        try:
            subprocess.run(["python", "poc.py"], check=True)
        except Exception as e:
            print(f"Failed to run poc.py: {e}")

    def update_frames(self):
        images = []
        try:
            for i, (pipeline, params) in enumerate(self.pipelines.items()):
                # 각 카메라에서 프레임 가져오기
                frames = pipeline.wait_for_frames() 
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                # RGB 프레임을 NumPy 배열로 변환
                color_image = np.asanyarray(color_frame.get_data())

                # Detect ArUco markers
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

                color_image_resized = cv2.resize(color_image, (640, 360))

                if ids is not None:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, params['camera_matrix'], params['dist_coeffs'])
                    params['rvec'] = rvecs[0].tolist()
                    params['tvec'] = tvecs[0].tolist()
                    R, _ = cv2.Rodrigues(rvecs[0])
                    # 변환 벡터를 3x1 행렬로 변환
                    T = tvecs[0].reshape((3, 1))
                    # 카메라의 월드 좌표 계산
                    camera_world_position = -np.dot(R.T, T).flatten()
                    color_image_resized = cv2.putText(color_image_resized, f"Camera position: {camera_world_position}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                # OpenCV 형식에서 QImage로 변환
                image = QImage(color_image_resized, color_image_resized.shape[1], color_image_resized.shape[0],
                            color_image_resized.strides[0], QImage.Format_BGR888)
                images.append(QPixmap.fromImage(image))
        except Exception as e:
            print(e)
        # QLabel에 QPixmap 설정
        for i in range(len(images)):
            self.labels[i].setPixmap(images[i])
    

    def closeEvent(self, event):
        # 프로그램 종료 시 모든 파이프라인 종료
        for pipeline in self.pipelines:
            pipeline.stop()
        event.accept()

    # TODO 1. poc.py 실행 버튼 만들기
    # TODO 2. 카메라 캘리브레이션 버튼 만들기
    # TODO 3. 카메라 영상에서 아루코 보드 인식 및 자체 캘리브레이션 결과가 카메라 영상 위에 나타나도록 박스 만들기

def main():
    app = QApplication(sys.argv)
    camWidget = CameraWidget(r"/home/zzol/FOCUS-1/modules/realsense/realsense_info.json")
    camWidget.setWindowTitle("RealSense Cameras")
    camWidget.show()
    sys.exit(app.exec_())
if __name__ == "__main__":
    main()