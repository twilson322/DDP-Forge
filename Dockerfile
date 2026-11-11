# C++ build + tests. The CasADi cross-check (scripts/crosscheck_casadi.py) runs on the host:
#   pip install casadi pandas matplotlib
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build libeigen3-dev libgtest-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
CMD ["bash", "-c", "cd build && ctest --output-on-failure"]
