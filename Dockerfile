FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libusb-1.0-0 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# selfpose3d_ui 리포지토리 클론
RUN git clone https://github.com/tabsanro/selfpose3d_ui.git /workspace/selfpose3d_ui

# 작업 디렉토리 설정
WORKDIR /workspace/selfpose3d_ui

# 필요한 Python 패키지 설치 및 모델 다운로드
RUN bash install.sh