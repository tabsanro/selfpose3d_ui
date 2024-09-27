- [FOCUS](#focus)
- [Quick Setup](#quick-setup)
- [conda setting](#conda-setting)
- [Record Dataset](#record-dataset)
- [Isaac Sim Built-in Python(3.10) Setting](#isaac-sim-built-in-python310-setting)
  - [Example](#example)
- [Directory look like](#directory-look-like)

# FOCUS
Framewise Observation and Continuous Update System - Digital Twin

# Quick Setup
To install the dependencies `gdown`, `requests`, and `tqdm` for `quick_setup.py` and then execute it, follow these steps:

1. Open a terminal.
2. Navigate to the directory where `quick_setup.py` is located.
3. Install the dependencies by running the following command:
  ``` sh
  pip install gdown requests tqdm ultralytics timm==0.8.13.dev0
  ```
4. Once the dependencies are installed, you can execute `quick_setup.py` by running the following command:
  ``` sh
  python quick_setup.py
  ```

By following these steps, you will install the 3rd-party dependencies for `quick_setup.py` and then execute it.

# conda setting
We recommend first one.
``` sh
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
or
``` sh
conda install pytorch==1.13.1 torchvision==0.14.1 cudatoolkit=11.7 -c pytorch -c nvidia
```

# Record Dataset
To record a dataset for your project, follow these steps:

1. Set up the necessary hardware and sensors for data collection.
2. Create a directory to store the recorded data. For example, you can use the command `mkdir dataset` to create a directory named "dataset".
3. Open a terminal and navigate to the directory where you want to store the recorded data. For example, you can use the command `cd dataset` to navigate to the "dataset" directory.
4. Run the recording script by executing the following command:
``` sh
python tools/record_dataset.py
```

In the `record_dataset.py` script, you can use the following command-line arguments:

- `--num-sources`: Specifies the number of cameras to record from. The default value is `4`.
- `--is-kafka`: Use confluent_kafka instead of local camera.
- `--duration`: Specifies the duration of the recording in seconds. The default value is `900` (15 minutes).
- `--imshow`: If provided, it will show the video feed during the recording.
- `--fourcc`: Specifies the FourCC codec to use for video recording. The default value is `mp4v`.
- `--fps`: Video frame rate. The default value is `30`.
- `--width`: Video size of `width`. The default value is `1920`.
- `--height`: Video size of `height`. The default value is `1080`
- `--device-list`: List of device numbers. This option is to specify the file name, like `hd_00_{num:02d}.mp4` If None, we are going to save the videos with file names like `hd_00_01`.
- `--server`: Kafka server

You can modify these arguments according to your requirements when running the `record_dataset.py` script.

1. This script will start recording data from the sensors and save it in the current directory.
2. During the recording process, perform the actions or scenarios you want to capture in your dataset.
3. Once you have finished recording, press a specific key or trigger an event to stop the recording.
4. The recorded data will be saved in the current directory in a format specified by the recording script.


# Isaac Sim Built-in Python(3.10) Setting
To register the `sim_python` of Isaac Sim in your `bashrc` file, you can follow these steps:

1. Open a terminal.
2. Open the `bashrc` file using a text editor. For example, you can use the command `vi ~/.bashrc`.
3. Scroll to the bottom of the file and add the following line:
   <!-- TODO 정확한 위치 알아내기 -->
    ```
    alias sim_python="/path/to/isaac_sim/python/kit/../python3"
    alias sim_python3='~/.local/share/ov/pkg/isaac-sim-2023.1.1/kit/python/bin/python3'
    ```
    Replace `/path/to/isaac_sim/python` with the actual path to the Python installation directory of Isaac Sim.
4. Save the changes and exit the text editor.
5. Run the command `source ~/.bashrc` to apply the changes to your current terminal session.

After following these steps, the `sim_python` of Isaac Sim should be registered in your `bashrc` file, allowing you to use it from any terminal session.

## Example
``` sh
sim_python -m pip list
```

# Directory look like
```
root/
|-- aisl/
|-- configs/
|   |-- cam4_posenet.yaml
|-- data/
|   |-- panoptic-toolbox/
|   |   |-- data/
|   |   |   |-- 16060224_haggling1/
|   |   |   |   |-- hdImgs/
|   |   |   |   |-- hdvideos/
|   |   |   |   |-- hdPose3d_stage1_coco19/
|   |   |   |   |-- calibration_160224_haggling1.json
|   |   |   |-- 160226_haggling1/
|   |   |   |-- ...
|   |   |   |-- group_train_cam5_pseudo_hrnet_soft_9videos.pkl
|   |   |   |-- group_train_cam5_pseudo_hrnet_hard_9videos.pkl
|   |   |   |-- group_validation_cam5_sub.pkl
|   |-- aisl/
|   |   |-- data/
|   |   |   |-- 24_07_05/
|   |   |   |-- hdVideos/
|   |   |   |   |-- hd_00_01.mp4
|   |   |   |   |-- hd_00_02.mp4
|   |   |   |   |-- hd_00_03.mp4
|   |   |   |   |-- hd_00_04.mp4
|   |   |   |-- calibrations/
|   |   |   |   |-- camera1.pkl
|   |   |   |   |-- camera2.pkl
|   |   |   |   |-- camera3.pkl
|   |   |   |   |-- camera4.pkl
|   |   |   |-- 240801/
|   |   |   |   ...
|-- SelfPose3d/
|   |-- config/
|   |-- lib/
|   |-- tools/
|   |-- ...
|-- mivolo/
|   |-- data/
|   |-- model/
|   |-- ...
|-- models/
|   |-- backbone_epoch20.pth.tar
|   |-- cam5_posenet.pth.tar
|   |-- cam5_rootnet_epoch2.pth.tar
|   |-- mivolo_imbd.pth.tar
|   |-- pose_resnet_50_384x288.pth
|   |-- yolov8x_person_face.pt
|-- tools/
|   |-- _init_paths.py
|   |-- record_dataset.py
|   |-- ...
```
