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
