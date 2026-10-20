#!/bin/bash
# Step 3/3: conda env "dreamer" with the official DreamerV3 (JAX 0.4.33, CUDA 12).
set -e
source ${AD_ROOT:-$HOME/airdreamer}/env.sh
conda create -y -p $Y/envs/dreamer python=3.11
conda activate $Y/envs/dreamer
cd $Y/src
[ -d dreamerv3 ] || git clone https://github.com/danijar/dreamerv3.git
cd dreamerv3
grep -v -E "^ale_py|^autorom" requirements.txt > $Y/cache/req_dreamer.txt   # Atari deps are not needed
pip install -r $Y/cache/req_dreamer.txt
python -c "import jax; print('JAX devices:', jax.devices())"
echo "DreamerV3 installed."
