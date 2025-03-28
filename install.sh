pip install --no-cache-dir -r requirements.txt
python modules/SelfPose3d/tools/download_from_drive.py

git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd torch2trt
python setup.py install

cd ..