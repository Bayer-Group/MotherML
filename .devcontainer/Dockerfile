# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
ARG VARIANT="3.13-trixie"
FROM mcr.microsoft.com/vscode/devcontainers/python:3-${VARIANT}

# [Choice] Node.js version: none, lts/*, 16, 14, 12, 10
ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi


# Install uv
RUN su vscode -c "umask 0002 && (curl -LsSf https://astral.sh/uv/install.sh | sh)"

#Install additional packages
# RUN sudo apt update && \
# sudo apt install libfontconfig1 libxrender1 libxtst6 -y

# [Optional] If your dependencies rarely change, uncomment this section to add them to the image.
RUN /home/vscode/.local/bin/uv sync --group test_duration
