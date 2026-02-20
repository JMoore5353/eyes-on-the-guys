FROM docker.io/osrf/ros:jazzy-desktop-full

# Update base system.
RUN apt-get update \
    && apt-get upgrade -y

# Create workspace directories.
RUN mkdir -p /rosflight_ws/src
WORKDIR /rosflight_ws/src

# Clone required repositories into /rosflight_ws/src.
RUN git clone -b v2.0.0 https://github.com/rosflight/rosflight_ros_pkgs --recursive \
    && git clone -b v2.0.0 https://github.com/rosflight/rosplane

WORKDIR /rosflight_ws

# Install rosdep dependencies.
RUN . /opt/ros/jazzy/setup.sh \
    && rosdep update \
    && rosdep install --from-paths . -y --ignore-src

# Build workspace.
RUN . /opt/ros/jazzy/setup.sh \
    && colcon build --symlink-install --executor sequential

# Copy this repository into /rosflight_ws/src/eyes-on-the-guys.
COPY . /rosflight_ws/src/eyes-on-the-guys

# Rosdep and build are repeated to avoid rebuilding rosflight_ros_pkgs and rosplane
# anytime the eyes-on-the-guys repository changes.
RUN . /opt/ros/jazzy/setup.sh \
    && rosdep update \
    && rosdep install --from-paths . -y --ignore-src
RUN . /opt/ros/jazzy/setup.sh \
    && colcon build --symlink-install --executor sequential

# Internal launch script used as the container entrypoint.
RUN cat <<'EOF' >/usr/local/bin/run-eyes-on-the-guys-sim.sh
#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
source /rosflight_ws/install/setup.bash

ros2 launch eyes_on_the_guys get_em.launch.py &
SIM_PID=$!

cleanup() {
    if ps -p "${SIM_PID}" >/dev/null 2>&1; then
        kill "${SIM_PID}" >/dev/null 2>&1 || true
        wait "${SIM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

sleep 10
ros2 launch rosflight_sim fixedwing_init_firmware.launch.py

ros2 service call /toggle_arm std_srvs/srv/Trigger
ros2 service call /toggle_override std_srvs/srv/Trigger

echo "!!! Simulation initialized. Press Ctrl+C to exit. !!!"
wait "${SIM_PID}"
EOF

# Make the entrypoint script executable.
RUN chmod +x /usr/local/bin/run-eyes-on-the-guys-sim.sh

# Set the entrypoint to the internal launch script.
ENTRYPOINT ["/usr/local/bin/run-eyes-on-the-guys-sim.sh"]
