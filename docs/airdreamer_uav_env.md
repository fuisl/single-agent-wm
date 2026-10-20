# Single-UAV Depth Navigation Environment (Isaac Sim 4.1 + OmniDrones)

A reinforcement-learning environment for single-drone navigation with a forward depth camera, following the
task setup of the AirDreamer paper (DreamerV3-based UAV navigation). A quadrotor has to fly from a start to a
goal through random obstacle fields (pillars, walls, U- and C-shaped traps) using only:

- a forward depth image, **48 x 80**, horizontal FOV 87° (RealSense D455-like), valid range **[0.6, 6] m**
  (out-of-range pixels are set to 6 m), and
- a 15-D body-frame state vector.

Physics runs at **200 Hz**, the policy acts at **20 Hz**. Many drones are simulated in parallel in one Isaac Sim
process, each in its own 24 x 24 m arena. A DreamerV3 (official JAX implementation) training and evaluation
setup is included.

## Requirements

- Linux (tested on Ubuntu 22.04), NVIDIA RTX GPU (tested on RTX 4090 24 GB)
- NVIDIA driver: tested with 530.30.02 (CUDA 12.1) for headless RTX rendering. NVIDIA recommends >= 535 for
  Isaac Sim 4.x; only the Iray offline renderer requires it, which is not used here
- About 30 GB disk space; no sudo or Docker needed
- Internet access to PyPI, pypi.nvidia.com and GitHub during installation

## Installation

All three scripts install under `AD_ROOT` (default `~/airdreamer`). Set it first if you want another location:

```bash
export AD_ROOT=/path/with/space/airdreamer   # optional
bash install/install_isaacsim.sh               # Miniforge, conda env "isaac41", Isaac Sim 4.1.0, PyTorch 2.2.2
bash install/install_isaaclab_omnidrones.sh    # Isaac Lab v1.1.0 + OmniDrones, plus compatibility fixes
bash install/install_dreamerv3.sh              # conda env "dreamer" with the official DreamerV3 (JAX)
```

`install_isaaclab_omnidrones.sh` applies three fixes that are needed for this version combination:

1. `boto3` / `botocore` pinned to 1.34.68, matching the copy bundled with Isaac Sim. Otherwise
   `omni.replicator` fails to load and cameras do not work
2. `tensordict==0.3.2`, required by `torchrl==0.3.1`, which OmniDrones uses
3. Removes `usd=usd` in Isaac Sim's `xform_prim_view.py` (from the OmniDrones troubleshooting page; the
   original file is kept as `.orig`)

Two conda envs are used because Isaac Sim 4.1 needs Python 3.10, while DreamerV3 uses Python 3.11 with JAX.

## Quick checks

```bash
source $AD_ROOT/env.sh && conda activate $AD_ROOT/envs/isaac41
python examples/smoke_test.py                 # headless render test, writes smoke.png
python examples/airdreamer_demo.py            # scripted flight demo video in outputs/demo/ (not a learned policy)
cd uav_env
python rl_server.py --num_envs 16 --port 6113 &
python bench_server.py --num_envs 16 --port 6113 --rounds 1000   # random-action throughput
```

On an otherwise idle RTX 4090, 16 drones run at about 300 agent-steps/s with random actions.

## Interface

| | Definition |
|---|---|
| `depth` | `uint8 (48, 80, 1)`: distance to image plane, linearly mapped from [0.6, 6] m to [0, 255] |
| `state` | `float32 (15,)`: goal direction in the yaw frame (3), horizontal and vertical goal distance (2), yaw-frame velocity (3), body angular velocity (3), orientation quaternion wxyz (4) |
| `action` | `float32 (5,)` in [-1, 1]: yaw-frame velocity command / Λ with Λ = diag(1.5, 1.5, 0.5) m/s (3), then cos and sin of the commanded absolute yaw (2). Tracked by OmniDrones' Lee position controller (velocity mode) |
| `reward` | 10·r_prog + 5·r_safety + 2·r_height + 100 on reaching the goal − 5 on collision, where r_prog is the decrease of the horizontal goal distance, r_safety = −max(0, 1 − d_obs)², and r_height = −(1.5 − z)² |
| termination | goal reached (within 1 m), collision (closer than 0.3 m to an obstacle), height outside [0.3, 3.2] m, or 60 s timeout (not terminal) |
| `log/*` | per-step outcome flags: goal, collision, height, timeout |

Obstacles are vertical boxes 3.5 m tall that are re-placed every episode. They only affect rendering;
collisions are checked analytically against the box geometry, so the drone's physics never touches them.

**Scenes** (`scenegen.py`): random arenas with 0–2 U/C traps, 1–3 walls and 6–16 pillars, plus seeded test
suites from `sample_test_scene`:

- `utrap`: start in front of a U trap's opening, goal behind it
- `utrap_in`: start inside a U trap facing its back wall
- `ctrap`: start inside a C trap (opening 1.2–1.8 m) facing its back wall

`rl_server.py --trap_start_frac 0.3` starts that fraction of training episodes inside a trap.

## Architecture

```
rl_server.py (isaac41 env)                         DreamerV3 (dreamer env)
  Isaac Sim, N drones, lock-step   <--- socket --->  N env processes (isaac_uav_env.IsaacUAV),
  ("step", action) / ("reset", opts)  per slot        one drone slot each
```

Every control step, each slot sends either `("step", action)` or `("reset", {"kind": ..., "seed": ...})`.
Once all N slots have sent a message, the server simulates 10 physics steps, renders once and answers each
slot with an observation dict. The protocol is plain `multiprocessing.connection`, so other RL libraries can
use the server through a small client like `isaac_uav_env.py` or `bench_server.py`.

## Training and evaluation with DreamerV3

```bash
cd uav_env
bash run_rl.sh runs/dreamer_uav --size size50m --steps 1.5e6 --train_ratio 64   # starts server + training
# optional curriculum: SERVER_ARGS="--trap_start_frac 0.3" bash run_rl.sh ...

source $AD_ROOT/env.sh; conda activate $AD_ROOT/envs/isaac41
python rl_server.py --num_envs 16 --port 6140 --extras &
conda activate $AD_ROOT/envs/dreamer
python eval_rl.py --logdir runs/dreamer_uav --port 6140 --episodes 100     # writes runs/dreamer_uav/eval/*.json
```

On one RTX 4090, the 50M model with 16 drones trains at about 90 environment steps/s, so 1.5M steps take
about 5 hours. JAX and Isaac Sim share the GPU; `MEMFRAC` (default 0.5) caps JAX's share.

## Known issues

- `PhysX error: PxArticulationLink::setGlobalPose(): it is illegal ...` is printed when drones are teleported
  at reset. It is harmless: teleports take effect
- With large, fast yaw changes the Lee controller can lose altitude; such episodes end with the `height` flag
- 16 environment processes connect at once at startup; the client retries for up to 60 s

## Credits

Built on NVIDIA Isaac Sim 4.1, Isaac Lab v1.1.0, OmniDrones (btx0424/OmniDrones) and DreamerV3
(danijar/dreamerv3). The task setup follows the AirDreamer paper.
