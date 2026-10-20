# LeWM for UAV depth navigation

[LeWorldModel](https://github.com/lucas-maes/le-wm) (JEPA, next-embedding prediction + SIGReg, no decoder,
no EMA) plugged into the same data, probe and Isaac test suites as the DreamerV3 world model. File layout follows
the official repo; `jepa.py` / `module.py` / `train.py` keep the official code and losses, without
`stable_worldmodel` / `stable_pretraining` (plain torch loop, our own npz episodes, a small CEM solver).

| file | |
|---|---|
| `module.py` | official SIGReg, ARPredictor (AdaLN-zero action conditioning), Embedder, MLP; + `vit_depth` (ViT-tiny on 1-channel 48x80, patch 8), `DepthDecoder`, `RewardHead` |
| `jepa.py` | JEPA: ViT CLS token + state embedding -> projector -> emb; predictor over `history_size` frames; reward-based `get_cost` |
| `train.py` | official `lejepa_forward` loss: `MSE(pred, next emb) + 0.09 * SIGReg(emb)`; hydra config `config/train/` |
| `train_heads.py` | post-hoc readouts on the frozen encoder: depth decoder (visualisation) + reward / terminal head (planning) |
| `extract.py` | per-step features for `probe/probe.py` + decoded reconstructions and open-loop imagination |
| `solver.py`, `eval.py` | CEM model-predictive control through `uav_env/rl_server.py`, replacing DreamerV3's actor; config `config/eval/` |

Differences from the official setup, all forced by the task:

- **Input**: 48x80 depth (1 channel, patch 8 -> 60 tokens) instead of 224x224 RGB; the 15-D body state
  (goal direction / distance, velocity, attitude) is embedded and added to the CLS token, since the goal is
  only given through the state.
- **Planning cost**: there is no goal image, so CEM maximises the return predicted by a reward / terminal head
  on the imagined embeddings (Dreamer-style continuation weighting) instead of `||z_T - z_goal||^2`.
- **Frameskip 5** (20 Hz -> 0.25 s blocks), history 3 frames, horizon 5 blocks, replan every block.

Memory: LeWM has no recurrent state. Everything it knows about the past is the predictor's context of
`history_size` frames (0.75 s at frameskip 5). `extract.py` exports both `ctx` (the whole window) and `emb`
(current frame only) so the probe can tell them apart.

## Run (main env, from `lewm/`)

```bash
source ../env.sh
$PY_MAIN train.py data.dir=$WM_ROOT/data/wm_v1 run_dir=$WM_ROOT/runs/lewm
$PY_MAIN train_heads.py --run_dir $WM_ROOT/runs/lewm --data $WM_ROOT/data/wm_v1
$PY_MAIN extract.py --run_dir $WM_ROOT/runs/lewm --data $WM_ROOT/data/wm_v1            # (+ --random_init)
$PY_MAIN ../probe/probe.py --data $WM_ROOT/data/wm_v1 --out $WM_ROOT/runs/probe \
    --feats lewm=$WM_ROOT/runs/lewm/features_ctx lewm_emb=$WM_ROOT/runs/lewm/features_emb wm=$WM_ROOT/runs/wm/features

$PY_ISAAC ../uav_env/rl_server.py --num_envs 16 --port 6140 --extras &
$PY_MAIN eval.py run_dir=$WM_ROOT/runs/lewm port=6140 episodes=100
```
