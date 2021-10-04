FROM nvcr.io/nvidia/pytorch:21.09-py3

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
  wget cmake ninja-build protobuf-compiler g++ \
  python3-dev libgflags-dev libgoogle-glog-dev libopencv-dev \
  gnupg2 software-properties-common git curl && \
  rm -rf /var/lib/apt/lists/*

RUN ln -sv /usr/bin/python3 /usr/bin/python

# checkout grpc
WORKDIR /opt
RUN git clone --depth 1 --branch v1.29.1 https://github.com/grpc/grpc.git
WORKDIR /opt/grpc
RUN git submodule update --init --recursive

# build the program:
RUN mkdir -p /opt/ml_service/build
COPY inference.cc /opt/ml_service
COPY inference.h /opt/ml_service
COPY sync_server.h /opt/ml_service
COPY async_server.h /opt/ml_service
COPY ocellus_ml_service.cc /opt/ml_service
COPY CMakeLists.txt /opt/ml_service
COPY ocellus_ml_service.proto /opt/ml_service

WORKDIR /opt/ml_service/build
ENV CPATH=/root/.local/include
ENV LIBRARY_PATH=/root/.local/lib
ENV LD_LIBRARY_PATH=/root/.local/lib
ENV CMAKE_PREFIX_PATH=/opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch/

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN cmake -DTORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST .. && make -j12

RUN curl -s https://packagecloud.io/install/repositories/immortal/immortal/script.deb.sh | bash
RUN apt install immortal

ENTRYPOINT immortal -l /var/log/ocellus_ml_service.log ./ocellus_ml_service
