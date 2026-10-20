# UAV World-Model Memory Probe Kit

Tools to measure how much a world model's latent state remembers about obstacles that have left the
camera's field of view, for the single-UAV depth-navigation task (Isaac Sim 4.1 + OmniDrones, AirDreamer setup).

Pipeline: **collect** flights with ground-truth maps → **train** a world model offline → **export** per-step
latents → **probe**: train a small decoder from the latent to an ego-centric occupancy map and score it on
unseen scenes, separately for visible / remembered / never-seen cells.

It is built for DreamerV3, but `probe.py` accepts latents from **any** world model (e.g. a JEPA-style model);
see "Using another world model" below.

## Requirements

Install the companion environment package (`airdreamer_uav_env`) first; it creates the `isaac41`
(Isaac Sim) and `dreamer` (DreamerV3/JAX) conda envs used here. Then copy this folder anywhere.

## 1. Collect data (isaac41 env)

```bash
python collect.py --num_envs 16 --episodes 1000 --out data/wm_v1          # ~1.5 h on an RTX 4090
python collect.py --num_envs 4 --episodes 8 --out data/wm_test --validate  # sanity check vs analytic depth
python viz_episode.py data/wm_v1/<episode>.npz                              # video of depth + labels + map
```

A scripted pilot (A* + pure pursuit + OU action noise) flies random arenas with pillars, walls and U/C traps:
70% fly straight to the goal, 15% explore random waypoints, and 15% enter a trap, look around and fly out.
Obstacles are rendered by Isaac Sim, and collisions are checked analytically. With `--validate`, rendered
depth matched analytic depth with a median error of 0 m.

Per episode, one `.npz` file holds per-step (20 Hz) arrays:

| key | shape | meaning |
|---|---|---|
| depth | (T,48,80) f16 | rendered depth, clipped to [0.6, 6] m |
| state | (T,15) | AirDreamer 15-D body-frame state |
| action | (T,5) | a_t chosen after seeing obs_t (DreamerV3 convention), 0 on the last step |
| reward, is_first, is_last, is_terminal | (T,) | |
| occ | (T,64,64) bool | ego-centric ground-truth occupancy, 0.15 m cells, row 0 = 4.8 m ahead, col 0 = 4.8 m left |
| seen | (T,64,64) bool | cell has been inside the (2D, occlusion-aware) camera view at least once this episode |
| vis | (T,64,64) bool | cell is visible right now |
| pos, yaw | (T,3), (T,) | arena-frame pose |

**Memory cells** = `occ & seen & ~vis`: obstacles that were seen earlier but are out of view now.

Train/val split: an episode is validation if `crc32(filename) % 1000 < 100` (about 10%), as in `probe.is_val`.

## 2. Train and export a DreamerV3 world model (dreamer env)

```bash
python train_wm.py train   --data data/wm_v1 --logdir runs/wm50m --size size50m --steps 20000   # ~35 min
python train_wm.py extract --data data/wm_v1 --logdir runs/wm50m --size size50m                  # -> runs/wm50m/features
python train_wm.py extract --data data/wm_v1 --logdir runs/wm50m --size size50m --random_init --feat_dir features_rand
```

`train` uses the unmodified DreamerV3 loss on random 64-step windows. `extract` replays whole episodes
(posterior, RSSM state carried across chunks, reset only at episode start) and saves `deter` (h_t), `stoch`
(z_t), the reconstruction, and an open-loop imagination (observe 32 steps, imagine 32 from actions only).

Auxiliary map loss: `--map_scale λ` adds a head decoding the occupancy map from (h_t, z_t), with BCE on
`seen` cells only. Implemented in `wm_agents.py`; `occ` and `seen` are never fed to the encoder or decoder.

## 3. Probe (isaac41 env, uses torch)

```bash
python probe.py --data data/wm_v1 --out runs/probe \
    --feats wm=runs/wm50m/features rand=runs/wm50m/features_rand
```

- Probe: MLP (LayerNorm → 1024 → 1024 → 64×64 logits, dropout 0.2), positive-weighted BCE (pos_weight 10),
  AdamW lr 3e-4 with weight decay 0.05, early stopping on 10% of the training episodes (patience 3)
- An extra `frame` baseline is a CNN probe on the current depth frame only, so it has no memory
- Metric: obstacle **average precision** (threshold-free) per cell group (visible now / memory / never seen),
  plus memory AP binned by steps since the cell was last visible (1–10, 11–20, 21–40, 41–80, 81–160, >160)
- Outputs: `probe_results.json`, `probe_summary.png`, `probe_examples.png`, `openloop_error.png`

Tip: probes overfit across scenes quickly (with about 800 training scenes), so keep early stopping and compare
models only within the same probe run.

## Using another world model (e.g. JEPA / LeWM)

For each episode file `ep_XXXXX_*.npz` in the dataset, write a file with **the same name** into a features
directory, containing:

- `feat`: float array of shape (T, D), where row t is the latent **after observing step t** (filtered over
  the whole episode from its first step), aligned with the dataset arrays
- optional `split`: "train" / "val"; if omitted, the crc32 rule above is used

Then pass the directory to `probe.py --feats jepa=<dir> ...`. Probing several models in one run, e.g.
`wm=... jepa=...`, makes them directly comparable.

## Our results so far (DreamerV3 50M, 1001 episodes, 20k offline updates)

Memory-cell AP on 88 unseen validation episodes (chance level ≈ 0.018):

| | memory AP |
|---|---|
| DreamerV3 world model (3 seeds) | 0.126 ± 0.006 |
| + map aux loss λ=10 (3 seeds) | 0.149 ± 0.002 |
| + map aux loss λ=30 (1 seed) | 0.165 |
| current-frame-only CNN | 0.072 |
| untrained world model | 0.045 |

Memory decays fast: the baseline drops from AP 0.35 (≤0.5 s out of view) to 0.13 (1–2 s) and 0.09 (2–4 s).
The map loss does not change depth reconstruction or open-loop prediction error. Details (in Chinese) and
figures are in `results/`.
