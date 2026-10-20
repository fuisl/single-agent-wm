#!/bin/bash
# Step 1/3: Miniforge + conda env "isaac41" with Isaac Sim 4.1.0 (pip) and PyTorch 2.2.2 (CUDA 12.1).
# Everything is installed under AD_ROOT (default ~/airdreamer); no sudo is needed.
set -e
export AD_ROOT=${AD_ROOT:-$HOME/airdreamer}
Y=$AD_ROOT
mkdir -p $Y/cache $Y/logs $Y/src
cat > $Y/env.sh <<EOT
export Y=$Y
export XDG_CACHE_HOME=\$Y/cache
export PIP_CACHE_DIR=\$Y/cache/pip
export TMPDIR=\$Y/cache/tmp
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
mkdir -p \$TMPDIR
source \$Y/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
EOT
# Optional (LINK_OV_CACHES=1): move Omniverse caches off a small home disk
if [ "${LINK_OV_CACHES:-0}" = 1 ]; then
  for d in .local/share/ov .cache/ov .nvidia-omniverse .nv; do
    if [ ! -e ~/$d ]; then mkdir -p $Y/home_links/$d ~/$(dirname $d); ln -s $Y/home_links/$d ~/$d; fi
  done
fi
cd $Y
if [ ! -d miniforge3 ]; then
  curl -sSL -o miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash miniforge.sh -b -p $Y/miniforge3
fi
source $Y/env.sh
conda create -y -p $Y/envs/isaac41 python=3.10
conda activate $Y/envs/isaac41
pip install --upgrade pip
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install isaacsim==4.1.0.0 isaacsim-extscache-physics==4.1.0.0 isaacsim-extscache-kit==4.1.0.0 \
    isaacsim-extscache-kit-sdk==4.1.0.0 --extra-index-url https://pypi.nvidia.com
# Isaac Sim bundles botocore 1.34.68; a newer boto3 pulled in by pip breaks omni.replicator
pip install "boto3==1.34.68" "botocore==1.34.68" "s3transfer<0.11"
echo "Isaac Sim installed. Next: bash install/install_isaaclab_omnidrones.sh"
