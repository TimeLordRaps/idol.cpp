FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG IDOL_SDK_REVISION=4702a7f00850e34b6e83a2170cc123c3e9570744

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake git ninja-build python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" IDOL_SDK=/opt/idol-sdk
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url "$TORCH_INDEX_URL" --extra-index-url https://pypi.org/simple
RUN git clone https://github.com/speridlabs/idol-sdk.git "$IDOL_SDK" \
    && git -C "$IDOL_SDK" checkout --detach "$IDOL_SDK_REVISION" \
    && pip install --no-cache-dir "$IDOL_SDK" gguf==0.19.0 accelerate==1.13.0 transformers==4.57.6 diffusers==0.36.0 \
       huggingface_hub==0.36.2 safetensors==0.8.0 tensordict==0.8.3 numpy==2.3.5 pillow==11.3.0 plyfile==1.1.5 scipy trimesh rtree==1.4.1

WORKDIR /opt/idol-cpp
COPY . /opt/idol-cpp
RUN pip install --no-cache-dir -r requirements-server.txt \
    && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build \
    && ctest --test-dir build --output-on-failure \
    && IDOL_MESH_EXE=/opt/idol-cpp/build/idol_mesh python -m unittest discover -s tests/python -p test_idol_server.py -v \
    && python -m unittest discover -s tests/python -p test_setup_iris.py -v \
    && python tests/python/test_python_install_audit.py \
    && python tools/audit_python_install.py --runtime-imports \
    && sync -f /opt/venv \
    && cmake --install build --prefix /usr/local

CMD ["bash"]
