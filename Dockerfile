FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
LABEL org.opencontainers.image.source=https://github.com/rhaps0dy/Automatic-Circuit-Discovery
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -q \
    && apt-get install -y --no-install-recommends \
    wget git git-lfs \
    python3 python3-dev python3-pip python3-setuptools python-is-python3 \
    libgl1-mesa-glx graphviz graphviz-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR "/Automatic-Circuit-Discovery"
COPY --chown=root:root requirements.txt ./
RUN pip install -r requirements.txt && rm -rf "${HOME}/.cache"

# Copy whole repo
COPY --chown=root:root . .
# Abort if repo is dirty
RUN if ! { [ -z "$(git status --porcelain --ignored=traditional)" ] \
    && [ -z "$(cd tracr && git status --porcelain --ignored=traditional)" ] \
    && [ -z "$(cd subnetwork-probing && git status --porcelain --ignored=traditional)" ] \
    ; }; then exit 1; fi
RUN pip install -e tracr -e . -e subnetwork-probing/transformer_lens && rm -rf "${HOME}/.cache"
