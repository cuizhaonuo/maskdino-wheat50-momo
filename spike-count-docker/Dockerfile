FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
ENV PATH=${CUDA_HOME}/bin:${CONDA_DIR}/bin:$PATH
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 \
    git \
    build-essential \
    gcc \
    g++ \
    make \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    ca-certificates \
    bzip2 \
    unzip \
    pkg-config \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniconda.sh && \
    conda clean -afy

SHELL ["/bin/bash", "-c"]

# 接受 Anaconda 默认 channels 的 ToS
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 创建 conda 环境并安装 PyTorch
RUN conda create --name maskdino python=3.8 -y && \
    echo "=== env created ===" && \
    conda run --no-capture-output -n maskdino conda install -y \
        pytorch==1.9.0 \
        torchvision==0.10.0 \
        cudatoolkit=11.1 \
        -c pytorch -c nvidia && \
    echo "=== pytorch installed ===" && \
    conda run --no-capture-output -n maskdino pip install --upgrade pip setuptools wheel && \
    echo "=== pip upgraded ===" && \
    conda run --no-capture-output -n maskdino pip install -U opencv-python

RUN conda run --no-capture-output -n maskdino conda install -y \
        "mkl<2024.1" \
        "intel-openmp<2024.1" \
        -c conda-forge && \
    echo "=== mkl patched ===" && \
    conda run --no-capture-output -n maskdino python -c "import torch; print(torch.__version__, torch.version.cuda)"

RUN mkdir -p /app
WORKDIR /app

# 先复制 requirements.txt，便于利用缓存
COPY requirements.txt /app/requirements.txt
RUN conda run -n maskdino pip install -r /app/requirements.txt

# 再复制整个项目
COPY . /app

RUN conda run --no-capture-output -n maskdino pip install detectron2==0.6 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# 安装额外依赖
# RUN conda run -n maskdino pip install git+https://github.com/cocodataset/panopticapi.git && \
#     conda run -n maskdino pip install git+https://github.com/mcordts/cityscapesScripts.git

COPY third_party/panopticapi /tmp/panopticapi
COPY third_party/cityscapesScripts /tmp/cityscapesScripts

RUN conda run --no-capture-output -n maskdino pip install /tmp/panopticapi && \
    conda run --no-capture-output -n maskdino pip install /tmp/cityscapesScripts

# 再执行一次项目依赖安装，确保复制完整项目后依赖一致
RUN conda run -n maskdino pip install -r /app/requirements.txt

# 编译自定义 CUDA/C++ 算子
RUN cd /app/maskdino/modeling/pixel_decoder/ops && \
    conda run -n maskdino bash make.sh

# 默认进入时激活环境
RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate maskdino" >> /root/.bashrc

WORKDIR /app

RUN cp /opt/conda/envs/maskdino/lib/python3.8/site-packages/detectron2/data/datasets/builtin_meta.py \
       /opt/conda/envs/maskdino/lib/python3.8/site-packages/detectron2/data/datasets/_builtin_meta.py

RUN conda run --no-capture-output -n maskdino pip install "setuptools<60"

RUN python - <<'PY'
from pathlib import Path

path = Path("/opt/conda/envs/maskdino/lib/python3.8/site-packages/yacs/config.py")
text = path.read_text()

old = 'raise KeyError("Non-existent config key: {}".format(full_key))'
new = 'print("Non-existent config key: {}".format(full_key))'

if old in text:
    text = text.replace(old, new)
    path.write_text(text)
    print("patched yacs config.py")
else:
    print("target not found, maybe already patched")
PY

RUN sed -i 's/raise KeyError("Non-existent config key: {}".format(full_key))/print("Non-existent config key: {}".format(full_key))/g' \
    /opt/conda/envs/maskdino/lib/python3.8/site-packages/yacs/config.py

CMD ["conda", "run", "--no-capture-output", "-n", "maskdino", "python", "main.py", "-i", "/data/images", "-o", "/output"]