#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

echo "== Serial devices visible on the host =="
ls -l /dev/serial/by-id 2>/dev/null || echo "No /dev/serial/by-id entries found."
serial_found=false
for device in /dev/ttyACM* /dev/ttyUSB*; do
  [[ -e "$device" ]] || continue
  ls -l "$device"
  serial_found=true
done
if [[ "$serial_found" == false ]]; then
  echo "No ttyACM/ttyUSB device found."
fi

echo
echo "== Video devices visible on the host =="
ls -l /dev/video* 2>/dev/null || echo "No /dev/video* device found."
for device in /dev/video*; do
  [[ -e "$device" ]] || continue
  echo "-- $device --"
  udevadm info --query=property --name="$device" 2>/dev/null \
    | grep -E '^(ID_V4L_PRODUCT|ID_SERIAL|ID_PATH|DEVNAME)=' || true
done

echo
echo "== Official LeRobot port finder =="
echo "Connect one arm at a time. The official tool will ask you to unplug it."
if [[ -t 0 ]]; then
  run_lerobot lerobot-find-port || true
else
  echo "Skipped because stdin is not an interactive terminal. Run this script from a terminal."
fi

echo
echo "== Official LeRobot OpenCV camera finder =="
run_lerobot lerobot-find-cameras opencv \
  --output-dir /workspace/outputs/captured_images

echo
echo "After this command, set FOLLOWER_PORT, LEADER_PORT, and CAMERA_TOP_DEVICE"
echo "in config/robot.env. Use CAMERA_WRIST_DEVICE for the optional second camera."
