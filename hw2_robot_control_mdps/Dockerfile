# Base image: NVIDIA CUDA 12.6.3 with cuDNN on Ubuntu 24.04
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Prevent apt from asking interactive questions during install
ENV DEBIAN_FRONTEND=noninteractive
# Set timezone to UTC
ENV TZ=UTC

# Install system packages and clean up apt cache to reduce image size
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    vim \
    htop \
    build-essential \
    cmake \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libegl1-mesa-dev \
    libglvnd-dev \
    && rm -rf /var/lib/apt/lists/*


# Make 'python' work as a command (Ubuntu 24.04 only ships with 'python3')
RUN ln -s /usr/bin/python3 /usr/bin/python

# Start in the home folder when the container launches
WORKDIR /home/tristan/robot_learning_course_2026/hw2_robot_control_mdps

# Drop into a bash shell by default when running the container
CMD ["/bin/bash"]
