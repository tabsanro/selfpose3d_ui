import pyrealsense2 as rs
import numpy as np
import json

def set_pipelines(webcam_info):
    with open(webcam_info) as f:
        camera_dict = json.load(f)
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