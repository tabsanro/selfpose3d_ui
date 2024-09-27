# # # !pip install pyrealsense2
import sys
import pyrealsense2 as rs
import numpy as np
import json
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QObject, QThread
import cv2
import subprocess
import os
import os.path as osp
import pickle

class RecordWidget(QWidget):
    def __init__(self, cam_info_path=None, path=None):
        super().__init__()
        self.move(0, 0)
        self.cam_info_path = cam_info_path
        self.path = path
        self.camera_dict = self.open_json(self.cam_info_path)
        self.pipelines = self.set_piplines(self.camera_dict)

        self.recording = False # 녹화 상태를 나타내는 플래그
        self.video_writers = {} # 비디오 기록자를 저장할 딕셔너리
        
        self.initUI()

        # 별도의 스레드에서 카메라 이미지를 지속적으로 받아옴
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker(self.pipelines, self.update_frames, self.save_frame)
        self.camera_worker.moveToThread(self.camera_thread)

        self.camera_thread.started.connect(self.camera_worker.process)
        self.camera_thread.start()

    def initUI(self):
        # 레이아웃 설정
        self.layout = QGridLayout()
        self.labels = []
        # 카메라의 수에 따라 QLabel 생성
        num_cameras = len(self.pipelines)
        self.cols = 2  # 기본적으로 두 열로 설정
        self.rows = (num_cameras + self.cols - 1) // self.cols

        self.record_button = QPushButton("Start Record", self)
        self.record_button.clicked.connect(self.toggle_recording)
        self.layout.addWidget(self.record_button, self.rows, 1)

        for i in range(num_cameras):
            label = QLabel(self)
            label.setScaledContents(True)
            self.labels.append(label)
            self.layout.addWidget(label, i // self.cols, i % self.cols)
        self.setLayout(self.layout)

    def toggle_recording(self):
        # 녹화 시작/중지 버튼
        self.recording = not self.recording
        if self.recording:
            self.start_recording()
            self.record_button.setText("Stop Record")
        else:
            self.stop_recording()
            self.record_button.setText("Start Record")

    def start_recording(self):
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(hdVideos:=osp.join(self.path, "hdVideos"), exist_ok=True)
        os.makedirs(calibration:=osp.join(self.path, "calibration"), exist_ok=True)
        # 비디오 기록자 설정 (1초에 15프레임)
        for i, (pipeline, params) in enumerate(self.pipelines.items()):
            filename = osp.join(hdVideos, f'hd_00_{i:02d}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
            self.video_writers[pipeline] = cv2.VideoWriter(filename, fourcc, 15.0, (1920, 1080))

            # 카메라 캘리브레이션 피클 저장
            with open(osp.join(calibration, f"camera{i}.pkl"), "wb") as f:
                pickle.dump(params, f)
        

    def stop_recording(self):
        # 비디오 기록자 종료
        for pipeline, writer in self.video_writers.items():
            if writer.isOpened():
                writer.release()  # 비디오 기록자 해제
        self.video_writers.clear()
    
    def save_frame(self, pipeline, frame):
        # 녹화 중일 때 프레임 저장
        if self.recording and pipeline in self.video_writers:
            try:
                self.video_writers[pipeline].write(frame)  # 프레임 기록
            except Exception as e:
                print(f"Error writing frame for pipeline {pipeline}: {e}")

    def open_json(self, path):
        # 카메라 내부 파라미터를 가져오기 위한 json 파일 열기
        with open(path) as f:
            return json.load(f)
  
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
                'rvec': np.array(camera['rvec']).T,
                'tvec': np.array(camera['tvec']).T
            }
            config = rs.config()
            config.enable_device(serial_number)
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
            pipeline.start(config)

        print("Connected to", len(pipelines), "cameras")
        return pipelines

    def update_frames(self, images):
        # 전달받은 이미지로 UI 업데이트
        for i in range(len(images)):
            self.labels[i].setPixmap(images[i])
    

    def closeEvent(self, event):
        # 프로그램 종료 시 모든 파이프라인 종료
        for pipeline in self.pipelines:
            pipeline.stop()
        if self.recording:
            self.stop_recording()
        self.camera_thread.quit()  # 스레드 종료
        self.camera_thread.wait()
        event.accept()

    # TODO 1. poc.py 실행 버튼 만들기
    # TODO 2. 카메라 캘리브레이션 버튼 만들기
    # TODO 3. 카메라 영상에서 아루코 보드 인식 및 자체 캘리브레이션 결과가 카메라 영상 위에 나타나도록 박스 만들기

class CameraWorker(QObject):
    def __init__(self, pipelines, update_callback, save_frame_callback):
        super().__init__()
        self.pipelines = pipelines
        self.update_callback = update_callback
        self.save_frame_callback = save_frame_callback

    def process(self):
        while True:
            images = []
            try:
                for pipeline, params in self.pipelines.items():
                    # 프레임이 준비될 때까지 기다림
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    
                    # RGB 프레임을 NumPy 배열로 변환
                    color_image = np.asanyarray(color_frame.get_data())

                    self.save_frame_callback(pipeline, color_image)

                    # 크기 조정
                    color_image_resized = cv2.resize(color_image, (640, 360))

                    # OpenCV 형식에서 QImage로 변환
                    image_qt = QImage(color_image_resized, color_image_resized.shape[1], color_image_resized.shape[0],
                                      color_image_resized.strides[0], QImage.Format_BGR888)
                    images.append(QPixmap.fromImage(image_qt))

                # 이미지가 준비되면 update_callback을 호출하여 UI 업데이트
                if images:
                    self.update_callback(images)

            except Exception as e:
                print(e)

def main():
    app = QApplication(sys.argv)
    camWidget = RecordWidget(r"/home/zzol/FOCUS-1/modules/realsense/realsense_info.json", r"/home/zzol/FOCUS-1/modules/SelfPose3d/data_temp")
    camWidget.setWindowTitle("RealSense Cameras")
    camWidget.show()
    sys.exit(app.exec_())
if __name__ == "__main__":
    main()