FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG IDOL_SDK_REVISION=4702a7f00850e34b6e83a2170cc123c3e9570744

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake git ninja-build python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" IDOL_SDK=/opt/idol-sdk
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url "$TORCH_INDEX_URL" --extra-index-url https://pypi.org/simple
RUN git clone https://github.com/speridlabs/idol-sdk.git "$IDOL_SDK" \
    && git -C "$IDOL_SDK" checkout --detach "$IDOL_SDK_REVISION" \
    && pip install --no-cache-dir "$IDOL_SDK" gguf==0.19.0 accelerate==1.13.0 transformers==4.57.6 diffusers==0.36.0 \
       huggingface_hub==0.36.2 safetensors==0.8.0 tensordict==0.8.3 numpy==2.3.5 pillow==11.3.0 plyfile==1.1.5 scipy trimesh rtree==1.4.1

ARG GGML_REVISION=57e2c2ca2a7cae756b1c4dfafa0cdfef9051d234
# The K-block quantization tools and their tests load a GGML shared library; v0.29.0 matches gguf-py 0.19.0.
RUN git clone https://github.com/ggml-org/ggml.git /tmp/ggml \
    && git -C /tmp/ggml checkout --detach "$GGML_REVISION" \
    && cmake -S /tmp/ggml -B /tmp/ggml/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
       -DGGML_NATIVE=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
    && cmake --build /tmp/ggml/build --target ggml-base \
    && mkdir -p /opt/ggml/lib \
    && cp -a /tmp/ggml/build/src/libggml-base.so* /opt/ggml/lib/ \
    && cp /tmp/ggml/LICENSE /opt/ggml/LICENSE \
    && rm -rf /tmp/ggml
ENV IDOL_GGML_LIB=/opt/ggml/lib/libggml-base.so

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
