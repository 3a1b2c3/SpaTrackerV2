# syntax=docker/dockerfile:1
#FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel AS base
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS base
#https://gitlab-master.nvidia.com/hpx/gfmlib/-/tree/main/tools/vault?ref_type=heads
#https://confluence.nvidia.com/pages/viewpage.action?pageId=1769137257
# Download and install the vault agent
#RUN curl -L https://urm.nvidia.com/artifactory/sw-kaizen-data-generic-local/com/nvidia/vault/vault-agent/2.3.1/nvault_agent_v2.3.1_linux_amd64.tar.gz | tar xz -C /usr/local/bin/
ENV USE_FLASH_ATTENTION=1
ENV TZ="Europe/London"
ARG DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
EXPOSE 6006
#ENV HF_DATASETS_OFFLINE=1 'local_files_only
#ENV HF_HUB_OFFLINE=1
ENV __NV_PRIME_RENDER_OFFLOAD=1
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility
ENV QT_X11_NO_MITSHM=1
ENV LIBGL_ALWAYS_INDIRECT=1
#ENV LIBGL_ALWAYS_SOFTWARE=1

#RUN groupadd -g 999 appuser && useradd -r -u 999 -g appuser appuser
#USER appuser
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        libglvnd0 \
        libgl1 \
        libglx0 \
        libegl1 \
        libgles2 && \
    rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        git \
        #swig \
        python3-dev \
        pkg-config \
        #libx11 \
        #libxext \
        xorg \
        xauth && \
    rm -rf /var/lib/apt/lists/*


#RUN python3 -m pip install --upgrade pip && pip install swig opencv-python --no-cache-dir \
RUN python3 -m pip install --upgrade pip && pip install opencv-python --no-cache-dir \
    opencv-fixer==0.2.5 && python -c "from opencv_fixer import AutoFix; AutoFix()"

RUN python3 -m pip install --verbose \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -i https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple/ \
        torchvision==0.18.1

RUN python3 -m pip install --verbose \
        gradio==5.31.0 \
        pydantic==2.10.6 \
        einops \
        einx \
        plotly \
        prettytable \
        decord \
        easydict \
        flow_vis \
        moviepy==1.0.0 \
        safetensors \
        scikit-learn \
        hydra-core \
        omegaconf \
        albumentations \
        matplotlib 

RUN python3 -m pip install --verbose \
        scikit-image \
        pycolmap==3.11.1 


RUN python3 -m pip install --verbose \
        git+https://github.com/facebookresearch/segment-anything.git \
        git+https://github.com/EasternJournalist/utils3d.git@d3a577acf0a9ad7e513a1416449a07b6f47d967f#egg=utils3d
RUN python3 -m pip install --verbose \
        huggingface_hub \
        "pyceres>=2.4" \
        "kornia>=0.8.1" \
        "xformers>=0.0.30" \
        timm \
        PyJWT \
        gdown \
        rich \
        decord \
        "ray[default]" \
        jaxtyping \
        transformers 

RUN python3 -m pip install --verbose \
        mediapy \
        rm -rf /var/lib/apt/lists/*
RUN mkdir data && mkdir config  && rm -rf /var/lib/apt/lists/*
ENV PYTHONPATH="/code/src"

FROM base AS dev
RUN apt-get update && apt-get install tk portaudio19-dev ffmpeg libsm6 libxext6  xdotool -y --no-install-recommends &&  rm -rf /var/lib/apt/lists/*
ADD tests ../tests
RUN mkdir code
WORKDIR /code
ADD tests ./tests
#ADD src ./src
#ENV CUDA_HOME="/usr/local/cuda-12.1"
#RUN python3 -m pip install flash_attn==2.5.9.post1 --use-pep517 --no-build-isolation
RUN apt update
RUN apt install wget

FROM base AS training
# adlrl cannnot find, useful for tensorboard on cluster but can be installed locally

# testing
RUN mkdir /code
WORKDIR /code
ADD src ./src

RUN mkdir data && rm -rf /var/lib/apt/lists/*
# tensorboard
ENV NCCL_DEBUG="INFO"
# original container symlinks /usr/lib/x86_64-linux-gnu/libcuda.so.1 -> libcuda.so.535.129.03
ENV LD_PRELOAD="/lib/x86_64-linux-gnu/libnvidia-ml.so.535.129.03:/lib/x86_64-linux-gnu/libcuda.so.535.129.03"
ENV NCCL_CUDA_PATH="/lib/x86_64-linux-gnu/libnvidia-ml.so.535.129.03:/lib/x86_64-linux-gnu/libcuda.so.535.129.03"
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.3cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
RUN pip install flash_attn-2.5.9.post1+cu122torch2.3cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
