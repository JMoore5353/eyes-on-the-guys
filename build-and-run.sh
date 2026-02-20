#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="eyes-on-the-guys-sim:jazzy"

if command -v podman >/dev/null 2>&1; then
    CONTAINER_CLI="podman"
elif command -v docker >/dev/null 2>&1; then
    CONTAINER_CLI="docker"
else
    echo "Error: neither podman nor docker is installed or on PATH."
    exit 1
fi

echo "Using container runtime: ${CONTAINER_CLI}"
echo "Building image ${IMAGE_NAME}..."
"${CONTAINER_CLI}" build -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"

XAUTH_FILE="${XAUTHORITY:-${HOME}/.Xauthority}"
RUN_ARGS=(
    run --rm -it --privileged
    -e "DISPLAY=${DISPLAY}"
    -e "QT_X11_NO_MITSHM=1"
    -e "XAUTHORITY=/tmp/.Xauthority"
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"
    -v "${XAUTH_FILE}:/tmp/.Xauthority:ro"
)

echo "Starting simulation container..."
"${CONTAINER_CLI}" "${RUN_ARGS[@]}" "${IMAGE_NAME}"
