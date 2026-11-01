# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
# FastWAM / LIBERO image for MAD. The evaluation image pin from the
# "Optimizing World Action Models on AMD GPUs" post, plus the
# faster-action-models tree and a gfx950 LIBERO simulator build.
# scripts/pyt_fam/run.sh calls the in-image Python entry points. It does
# not run docker, so this image is the only container madengine starts.

ARG BASE_DOCKER=rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0
FROM $BASE_DOCKER
USER root

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    PATH="/opt/venv/bin:${PATH}" \
    FAM_ROOT=/opt/faster-action-models \
    LIBERO_ROOT=/opt/LIBERO \
    LIBERO_CONFIG_PATH=/opt/libero_config \
    INSTINCT_SIM_BUILD=/opt/faster-action-models/simulate/build-libero \
    HIP_ARCH=gfx950

ARG FAM_COMMIT=a5fa73101988870e51f04fe1e7be5da56dda3841
ARG LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libegl-mesa0 libosmesa6 libglu1-mesa \
        git ca-certificates time cmake ninja-build build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/AMD-AGI/faster-action-models.git /opt/faster-action-models \
 && git -C /opt/faster-action-models checkout -q "${FAM_COMMIT}"

# Pinned evaluator stack from docker/Dockerfile.eval. fsspec is installed
# without the resolver because it is newer than the range datasets declares.
RUN python3 -c "import torch; print('base torch', torch.__version__)" \
 && grep -v '^fsspec==' /opt/faster-action-models/eval/native/requirements-fastwam-libero.txt > /tmp/requirements-no-fsspec.txt \
 && pip install -r /tmp/requirements-no-fsspec.txt \
 && pip install --no-deps "$(grep '^fsspec==' /opt/faster-action-models/eval/native/requirements-fastwam-libero.txt)"

RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO /opt/LIBERO \
 && git -C /opt/LIBERO checkout -q "${LIBERO_COMMIT}" \
 && pip install --no-deps -e /opt/LIBERO \
 && echo /opt/LIBERO > "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/libero_checkout.pth"

RUN mkdir -p /opt/libero_config /opt/LIBERO/libero/datasets && L=/opt/LIBERO/libero/libero && printf '%s\n' \
      "benchmark_root: $L" "bddl_files: $L/bddl_files" "init_states: $L/init_files" \
      "datasets: /opt/LIBERO/libero/datasets" "assets: $L/assets" > /opt/libero_config/config.yaml \
 && chmod -R a+rX /opt/libero_config \
 && cd / && python3 -c "from libero.libero import benchmark, get_libero_path; print('libero', get_libero_path('bddl_files'))" < /dev/null

# gfx950 simulator (MI350X / MI355X). rocm-sdk-devel supplies the HIP
# compiler the LIBERO preset links against. See simulate/docker/Dockerfile.build.
RUN pip install --index-url https://repo.amd.com/rocm/whl-multi-arch 'rocm-sdk-devel==7.14.0' \
 && pip install pybind11 \
 && (command -v rocm-sdk >/dev/null && rocm-sdk init || true) \
 && cd /opt/faster-action-models && JOBS=8 bash simulate/build.sh libero \
 && ls /opt/faster-action-models/simulate/build-libero/python/instinct_sim*.so

RUN python3 -c "import torch, torchvision; v = torch.__version__; print('torch', v, 'torchvision', torchvision.__version__); assert v.startswith('2.11.0+rocm7.14'), v"
