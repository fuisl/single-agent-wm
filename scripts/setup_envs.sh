#!/bin/bash
# Creates the three uv environments (no sudo).  bash scripts/setup_envs.sh [main|isaac|dreamer ...]
# Replaces docs/original_install_scripts (conda) with uv; same pins and the same three compatibility fixes.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
mkdir -p third_party
TARGETS=${*:-main isaac dreamer}

clone() {  # clone <url> <dir> [branch]
  [ -d third_party/$2 ] || git clone -q ${3:+--depth 1 -b $3} $1 third_party/$2
}

setup_main() {
  uv sync
  $PY_MAIN -c "import torch; print('main torch', torch.__version__, torch.cuda.is_available())"
}

setup_isaac() {
  clone https://github.com/isaac-sim/IsaacLab.git IsaacLab v1.1.0
  clone https://github.com/btx0424/OmniDrones.git OmniDrones
  uv venv --seed --python 3.10 .venvs/isaac
  export VIRTUAL_ENV=$WM_ROOT/.venvs/isaac
  uv pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
  uv pip install --index-strategy unsafe-best-match --extra-index-url https://pypi.nvidia.com \
    isaacsim==4.1.0.0 isaacsim-extscache-physics==4.1.0.0 isaacsim-extscache-kit==4.1.0.0 \
    isaacsim-extscache-kit-sdk==4.1.0.0 torch==2.2.2
  uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.2 cmake usd-core==23.11 lxml==4.9.4 tqdm xxhash opencv-python-headless scipy \
    -e third_party/IsaacLab/source/extensions/omni.isaac.lab \
    -e third_party/IsaacLab/source/extensions/omni.isaac.lab_assets \
    -e third_party/IsaacLab/source/extensions/omni.isaac.lab_tasks \
    -e third_party/OmniDrones
  # fix 1: Isaac Sim bundles botocore 1.34.68; a newer boto3 breaks omni.replicator (no cameras)
  # fix 2: torchrl 0.3.1 (OmniDrones) needs tensordict 0.3.2
  uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.2 "boto3==1.34.68" "botocore==1.34.68" "s3transfer<0.11" "tensordict==0.3.2" "numpy<2"
  # fix 3: OmniDrones troubleshooting, drop usd=usd in xform_prim_view.py (original kept as .orig)
  F=$VIRTUAL_ENV/lib/python3.10/site-packages/isaacsim/exts/omni.isaac.core/omni/isaac/core/prims/xform_prim_view.py
  cp -n $F $F.orig && sed -i "s/self.get_world_poses(usd=usd)/self.get_world_poses()/" $F
  $PY_ISAAC -c "import torch, torchrl, tensordict; print('isaac', torch.__version__, torchrl.__version__, tensordict.__version__, torch.cuda.is_available())"
  unset VIRTUAL_ENV
}

setup_dreamer() {
  clone https://github.com/danijar/dreamerv3.git dreamerv3
  uv venv --python 3.11 .venvs/dreamer
  grep -v -E "^ale_py|^autorom" third_party/dreamerv3/requirements.txt > .venvs/dreamer/req.txt   # no Atari
  VIRTUAL_ENV=$WM_ROOT/.venvs/dreamer uv pip install -r .venvs/dreamer/req.txt ruamel.yaml
  $PY_DREAMER -c "import jax; print('dreamer JAX devices:', jax.devices())"
}

for t in $TARGETS; do echo "=== setup $t"; setup_$t; done
echo SETUP_ENVS_DONE
