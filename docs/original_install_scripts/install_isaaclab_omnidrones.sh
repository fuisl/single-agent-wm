set -e
# Step 2/3: Isaac Lab v1.1.0 + OmniDrones in the isaac41 env, plus the required compatibility fixes.
# 版本固定：tensordict 需与 torchrl 0.3.1 匹配
source ${AD_ROOT:-$HOME/airdreamer}/env.sh
conda activate $Y/envs/isaac41
cd $Y/src
[ -d IsaacLab ] || git clone -q --depth 1 -b v1.1.0 https://github.com/isaac-sim/IsaacLab.git
pip install cmake usd-core==23.11 lxml==4.9.4 tqdm xxhash
cd IsaacLab
./isaaclab.sh --install none
cd $Y/src/OmniDrones
pip install -e .
pip install "boto3==1.34.68" "botocore==1.34.68" "s3transfer<0.11"
pip install "tensordict==0.3.2"
F=$CONDA_PREFIX/lib/python3.10/site-packages/isaacsim/exts/omni.isaac.core/omni/isaac/core/prims/xform_prim_view.py
cp -n $F $F.orig && sed -i "s/self.get_world_poses(usd=usd)/self.get_world_poses()/" $F
python -c "import torch, torchrl, tensordict; print('VERS', torch.__version__, torchrl.__version__, tensordict.__version__, torch.cuda.is_available())"
echo SETUP2_DONE
