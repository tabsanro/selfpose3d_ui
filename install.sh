#/bin/bash
apt update
apt install -y libfreetype6 libpng16-16 libjpeg8 libtiff5 libopenjp2-7 libimagequant0 libwebp7 libxcb1
pip install -r requirements_jetson.txt

python3 modules/SelfPose3d/tools/download_from_drive.py

git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd torch2trt
python3 setup.py install
cd ..
rm -rf torch2trt