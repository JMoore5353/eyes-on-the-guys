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
COPY run-eyes-on-the-guys-sim.sh /usr/local/bin/
# Make the entrypoint script executable.
RUN chmod +x /usr/local/bin/run-eyes-on-the-guys-sim.sh

# Set the entrypoint to the internal launch script.
ENTRYPOINT ["/usr/local/bin/run-eyes-on-the-guys-sim.sh"]
