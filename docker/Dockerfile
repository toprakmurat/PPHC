FROM ubuntu:22.04

LABEL maintainer="SecurePulse-FHE Team"
LABEL description="Privacy-Preserving Health Analysis with FHE"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    libgmp-dev \
    libntl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy scikit-learn

WORKDIR /app
COPY . .

RUN mkdir -p build && cd build && \
    cmake -GNinja -DCMAKE_BUILD_TYPE=Release .. && \
    ninja

EXPOSE 8443
CMD ["./build/securepulse_server"]
