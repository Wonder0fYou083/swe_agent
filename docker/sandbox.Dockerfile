# The sandbox image. Every `bash` command the agent runs starts a fresh container from it.
#
# Build it once. This needs the network, to pull the base image and pytest:
#   docker build -t swe-agent-sandbox -f docker/sandbox.Dockerfile docker/
#
# Containers run with --network none, so nothing can be installed at run time:
# whatever the agent needs must be in this image.
FROM python:3.12-slim

# The same pytest version as uv.lock, so the agent's test runs match the host's.
RUN pip install --no-cache-dir pytest==9.1.1

# The repo is mounted here. Containers run as the host user, who has no home in the
# image, so the run sets HOME=/tmp.
WORKDIR /workspace
