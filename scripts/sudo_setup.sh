#!/bin/bash
# One-time machine setup that needs root. Run once:  sudo bash scripts/sudo_setup.sh
# Undo the sleep part with:  sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target
set -e
USER_NAME=${SUDO_USER:-fuisloy}

# 1. ffmpeg (episode / demo videos), GLU + OpenGL (Isaac Sim fails to start without them), vulkan-tools (Isaac Sim rendering, `vulkaninfo` for debugging)
apt-get update
apt-get install -y ffmpeg libglu1-mesa libopengl0 vulkan-tools

# 2. +16 GB swap: Isaac Sim alone takes ~8 GB RAM and this machine has 14 GB
if [ ! -f /swapfile_wm ]; then
  fallocate -l 16G /swapfile_wm
  chmod 600 /swapfile_wm
  mkswap /swapfile_wm
  echo '/swapfile_wm none swap sw 0 0' >> /etc/fstab
fi
swapon /swapfile_wm 2>/dev/null || true

# 3. Never suspend (lid close / idle) so overnight runs survive; keep the laptop on AC power
systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

# 4. Keep user processes and `systemd --user` units alive after logout
loginctl enable-linger "$USER_NAME"

swapon --show
echo SUDO_SETUP_DONE
