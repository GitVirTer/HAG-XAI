FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

WORKDIR /home/lwf/PD/FullGradCAM_Yolov5s

COPY . /home/lwf/PD/FullGradCAM_Yolov5s/

RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update \
    && pip config set global.indexi-url https://mirrors.aliyun.com/pypi/simple \
    && apt install -y libgl1-mesa-glx \
    && apt-get install -y libglib2.0-dev \
    && pip install gitpython>=3.1.30 \
    && pip install pandas==2.0.0 \
    && pip install seaborn==0.12.2 \
    matplotlib>=3.3 \
    numpy>=1.18.5 \
    opencv-python==4.6.0.66 \
    Pillow \
    psutil \
    PyYAML>=5.3.1 \
    requests>=2.23.0 \
    scipy>=1.4.1 \
    thop>=0.1.1 \
    torch>=1.7.0 \
    torchvision>=0.8.1 \
    tqdm>=4.64.0 \
    setuptools>=65.5.1 \
    prefetch_generator \
    && pip install -r requirements.txt
