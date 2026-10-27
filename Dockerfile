# Training and inference environment (the challenge runtime: Python 3.11, PyTorch 2.4.1, CUDA 12.1).
#   docker build -t upar2027-oimook .
#   docker run --gpus all --shm-size 16g -v $PWD/data:/workspace/data -v $PWD/outputs:/workspace/outputs \
#       -it upar2027-oimook bash
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace
COPY requirements.txt .
# torch and torchvision come with the image; the remaining packages follow requirements.txt.
RUN grep -v -E '^(--extra-index-url|torch==|torchvision==)' requirements.txt > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt
COPY . .
