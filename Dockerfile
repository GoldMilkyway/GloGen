# Base image with PyTorch, CUDA, and cuDNN
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Create a working directory
RUN mkdir /bp_benchmark
COPY ./ /bp_benchmark
WORKDIR /bp_benchmark

# Update apt and install necessary packages, then clean up cache
RUN apt-get update -y && apt-get install -y \
    git \
    tmux \
    wget \
    unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in one go and avoid cache
RUN pip install --no-cache-dir Cython==0.29.26 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir git+https://github.com/cainmagi/MDNC.git \
    && pip install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --no-cache-dir --upgrade requests \
    && pip install --no-cache-dir numpy==1.20.0 \
    && pip install --no-cache-dir hydra-optuna-sweeper==1.2.0 \
    && pip install --no-cache-dir wandb

# Set the default command to run on container start
CMD ["bash"]