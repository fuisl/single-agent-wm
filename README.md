# single-agent-wm

How much does a world model remember about obstacles that left the camera view, and does that memory help a
single UAV navigate (escape U/C traps) with only a forward depth camera? Setup follows AirDreamer
(DreamerV3-based UAV navigation): Isaac Sim 4.1 + OmniDrones, 48x80 depth, 15-D state, 5-D action, 20 Hz.

## Layout

| dir | what | env |
|---|---|---|
| `uav_env/` | Isaac server for RL (`rl_server.py`), data collection with a scripted pilot (`collect.py`), scenes / A* / labels (`scenegen.py`), viz | `.venvs/isaac` |
| `dreamer/` | DreamerV3: offline WM training + feature export (`train_wm.py`), online RL (`train_rl.py`, `run_rl.sh`), eval (`eval_rl.py`) | `.venvs/dreamer` |
| `probe/` | memory probe: latent -> ego occupancy map, AP on visible / memory / never-seen cells | `.venv` |
| `scripts/` | env setup (`setup_envs.sh`), one-time root setup (`sudo_setup.sh`), pipelines | |
| `docs/` | original READMEs of the two kits, earlier DreamerV3 results (`docs/dreamer_results/`) | |

## Setup

```bash
sudo bash scripts/sudo_setup.sh     # once: ffmpeg/GLU, extra swap, no suspend, linger
bash scripts/setup_envs.sh          # uv envs: .venv (main), .venvs/isaac (py3.10), .venvs/dreamer (py3.11)
source env.sh                       # PYTHONPATH, DV3_ROOT, $PY_MAIN / $PY_ISAAC / $PY_DREAMER
```

## Pipeline

```bash
$PY_ISAAC uav_env/collect.py --num_envs 8 --episodes 1000 --out data/wm_v1
$PY_DREAMER dreamer/train_wm.py train   --data data/wm_v1 --logdir runs/wm --size size50m --steps 20000
$PY_DREAMER dreamer/train_wm.py extract --data data/wm_v1 --logdir runs/wm --size size50m
$PY_MAIN probe/probe.py --data data/wm_v1 --out runs/probe --feats wm=runs/wm/features
```

Online RL: `bash dreamer/run_rl.sh runs/rl --size size50m --steps 1.5e6`. See `docs/` for details.
