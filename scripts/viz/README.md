# Visualisations (main env: `source env.sh`, then `$PY_MAIN ...`; `--device cpu` is the default where a model runs)

| script | makes | command |
|---|---|---|
| `render_episode.py` | `runs/kin_v1/viz/episodes/`: per-episode MP4 + PNGs: top-down world (visible / remembered obstacle cells, field of view), chase-camera 3-D render, onboard depth, depth decoded from LeWM emb / CLS / patch tokens, patch-token PCA, ego occupancy truth vs probe maps | `scripts/viz/render_episode.py [--episodes ep_....npz] [--steps 120:420] [--png 200,300] [--probe_sets wm lewm_emb lewm_patch frame m2_belief] --device cpu` |
| `latent_figures.py` | `runs/kin_v1/viz/figures/`: `latent_pca.png`, `decoders.png`, `memory_demand.png`, `probe_groups.png`, `probe_age.png` (official LeWM diagnosis) | `scripts/viz/latent_figures.py` |
| `render_memtest.py` | `runs/kin_v1/viz/memtest/`: the 4 twin variants (seen / novel / removed / absent) of one look-away scenario side by side (chase camera, wall W in blue or as a dashed ghost, onboard depth) + each model's latent prediction error over time, re-entry frames shaded | `scripts/viz/render_memtest.py --scenario 0 --gap 40 --run_dirs runs/kin_v1/lewm runs/kin_v1/m2 runs/kin_v1/m4` |
| `memory_figures.py` | `runs/kin_v1/viz/figures/`: `mem_retention.png` (E1 vs time out of view), `mem_surprise.png` (E2 + VoE accuracy), `mem_lesions.png` (E3 swap / reset), `mem_traces.png` (error around the turn-back), `mem_probe_age.png` (probe memory AP vs age, LeWM ladder vs DreamerV3) | `scripts/viz/memory_figures.py` |
| `belief_figures.py` | `runs/kin_v1/viz/belief/`: `traj`: belief PCA trajectory over a held-out trap episode with leave / re-enter markers; `map`: M4 latent-map video (PCA RGB + energy) next to the ego truth | `scripts/viz/belief_figures.py traj` · `scripts/viz/belief_figures.py map --memtest data/memtest_v1/sc_000_gap040_seen.npz` · `... map --episode ep_00060_kin_trap_goal.npz` |

- `vizlib.py` (ray-cast chase camera, box outlines, ego colours, labels) and `lewmlib.py` (load a LeWM / R-LeWM
  checkpoint, embed a rollout, one-step errors as in `lewm/memtest.py`, beliefs / M4 maps) are shared helpers.
  `lewmlib.py` only reads `lewm/`.
- Missing optional inputs are skipped with a note. Probe feature sets `m<i>_<part>` and run dirs `m<i>` (or
  `smoke_m<i>`) are labelled automatically.
- `runs/kin_v1/viz/mt_small/` is a 2-scenario memtest set for testing (`--data runs/kin_v1/viz/mt_small`).
- CPU cost: ~2.5 min per 300 episode steps (`render_episode`), ~3 min per twin video with 3 models (`render_memtest`),
  < 1 min per map video.
