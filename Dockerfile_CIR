FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
    apt-get install -y \
    gcc git wget \
    ffmpeg libsm6 libxext6 \
    python3.8 python3.8-dev python3-pip
    
RUN pip install torch==1.11.0 torchvision torchaudio fvcore --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
RUN pip install wandb sklearn scikit-image ipython ninja pandas opencv-python tqdm
RUN git clone https://github.com/nadeemlab/CIR.git 
WORKDIR CIR
RUN git submodule init && git submodule update
RUN wget -q https://zenodo.org/record/6762573/files/CIRDataset_pickle_for_voxel2mesh.tar.bz2 \
    && wget -q https://zenodo.org/record/6762573/files/pretrained_model-mesh%2Bencoder.tar.bz2 \
    && wget -q https://zenodo.org/record/6762573/files/pretrained_model-meshonly.tar.bz2 \
    && tar xjf CIRDataset_pickle_for_voxel2mesh.tar.bz2 \
    && tar xjf pretrained_model-mesh+encoder.tar.bz2 \
    && tar xjf pretrained_model-meshonly.tar.bz2 && rm *.bz2
