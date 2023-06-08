FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
LABEL org.opencontainers.image.source=https://github.com/arthurconmy/Automatic-Circuit-Discovery
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -q \
    && apt-get install -y --no-install-recommends \
    wget git git-lfs \
    python3 python3-dev python3-pip python3-venv python3-setuptools python-is-python3 \
    libgl1-mesa-glx graphviz graphviz-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# This venv only holds Poetry and its dependencies. They are isolated from the main project dependencies.
ENV POETRY_HOME="/opt/poetry"
RUN python3 -m venv $POETRY_HOME \
    # Here we use the pip inside $POETRY_HOME but afterwards we should not
    && "$POETRY_HOME/bin/pip" install poetry==1.4.2 \
    && rm -rf "${HOME}/.cache"
ENV POETRY="${POETRY_HOME}/bin/poetry"

WORKDIR "/Automatic-Circuit-Discovery"
COPY --chown=root:root pyproject.toml poetry.lock ./

# Don't create a virtualenv, the Docker container is already enough isolation
RUN "$POETRY" config virtualenvs.create false \
    # Install dependencies
    && "$POETRY" install --no-root --no-interaction "--only=main,dev" \
    && rm -rf "${HOME}/.cache"

# Copy whole repo
COPY --chown=root:root . .
# Abort if repo is dirty
RUN if ! { [ -z "$(git status --porcelain --ignored=traditional)" ] \
    ; }; then exit 1; fi

# Finally install this package
RUN "$POETRY" install --only-root --no-interaction
