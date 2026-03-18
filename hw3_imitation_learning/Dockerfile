# Base image: NVIDIA CUDA 12.6.3 with cuDNN on Ubuntu 24.04
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
# Prevent apt from asking interactive questions during install
ENV DEBIAN_FRONTEND=noninteractive
# Set timezone to UTC
ENV TZ=UTC
# tells MuJoCo to use EGL for headless GPU rendering
ENV MUJOCO_GL=egl

# switch from /bin/sh to /bin/bash
SHELL ["/bin/bash", "-c"]

# Install system packages and clean up apt cache to reduce image size
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    git \
    vim \
    nano \
    htop \
    tmux \
    unzip \
    zip \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libgl1 \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    libglfw3-dev \
    libglew-dev \
    libglib2.0-0 \
    libgomp1 \
    net-tools \
    iputils-ping \
    ca-certificates \
    locales \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Make 'python' work as a command (Ubuntu 24.04 only ships with 'python3')
RUN ln -s /usr/bin/python3 /usr/bin/python

# Rename existing UID/GID 1000 user to tristan
RUN usermod -l tristan $(getent passwd 1000 | cut -d: -f1) && \
    groupmod -n tristan $(getent group 1000 | cut -d: -f1) && \
    usermod -d /home/tristan -m tristan && \
    echo "tristan ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to tristan user
USER tristan

# Set working directly via argument
ARG WORKDIR
WORKDIR $WORKDIR

# Set up hw3_imitation_learning venv
#RUN uv venv --python 3.12 \
#    && source .venv/bin/activate \
#    && uv pip install -e .

# Drop into a bash shell by default when running the container
CMD ["/bin/bash"]

# docker build --build-arg WORKDIR=$(pwd) -t your_image .
# docker run -it --gpus all -v /home/tristan:/home/tristan your_image
