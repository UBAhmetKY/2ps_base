FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        g++ \
        gcc \
        make \
        pkg-config \
        wget \
        git \
        libgoogle-glog-dev \
        libgflags-dev \
        libboost-all-dev \
        libopenmpi-dev \
        openmpi-bin \
        libsparsehash-dev \
        ca-certificates \
        libgomp1 \
        libgoogle-glog0v5 \
        libgflags2.2 \
        libboost-mpi1.74.0 \
        libopenmpi3 \
        python3 \
        python3-pip && \
    python3 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install "numpy<2.0" && \
    pip install torch==2.0.1 torchdata==0.6.1 dgl ogb PyYAML pydantic && \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip

# CMake 3.26.4 installieren
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh && \
    sh cmake-3.26.4-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.26.4-linux-x86_64.sh

WORKDIR /app

# Build ausführen
RUN --mount=type=bind,source=.,target=/sources,readonly=0 \
    sh -c " cd /sources && \
            mkdir build && \
            cd build && \
            cmake .. && \
            make -j\$(nproc) && \
            cp twophasepartitioner /app/ && \
            cp ../*.py /app/"


CMD ["/bin/bash"]


