#!/bin/bash
# Optional, needs a reboot. Isaac Sim 4.1's RTX renderer segfaults on NVIDIA driver 595 (in
# omni.kit.widget.viewport __enable_hydra_engine, any time rendering starts); 4.1 was validated on 535-550.
#   sudo bash scripts/sudo_driver.sh [580|570|550]     then reboot, then: bash scripts/quick_test.sh isaac
# 580 keeps CUDA 13 for the main env's torch (cu130). With 570/550 the main env needs a cu12x torch.
# Revert: sudo bash scripts/sudo_driver.sh 595
set -e
V=${1:-580}
apt-get update
apt-get install -y nvidia-driver-$V
apt-mark hold nvidia-driver-$V        # stop unattended upgrades from moving it back to 595
echo "Installed nvidia-driver-$V. Reboot now."
