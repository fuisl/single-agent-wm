# LeWM and obstacle memory: literature review and failure analysis (run `kin_v1`)

Date: 2026-10-21. Data: `data/kin_v1` (1000 kinematic stand-in episodes, scripted goal / trap / explore policy).
Runs: `runs/kin_v1/{lewm,wm,probe,probe_untrained}`. Diagnostics scripts: `scripts/diag/`.

## 0. Summary

- **LeWM is not "broken" in the sense of a coding bug.** The training loop, loss, predictor, projector and
  SIGReg match the official `lucas-maes/le-wm` code. The one structural deviation is ours by design: the 15-D body
  state is *summed into the CLS token* before the projector (official LeWM is pixels-only).
- **The memory question cannot be answered with LeWM as specified, by construction and by objective.**
  The latent is a per-frame embedding, the predictor sees 3 frames (0.75 s), and the training signal for memory is
  almost absent in our data: for the next predicted frame, only **1.0 %** of the visible obstacle content was last
  seen more than 0.75 s earlier (91 % is in the 3 context frames, 6.7 % is new). With 1-step teacher-forced
  prediction the gradient towards remembering is ~0 (Gregor et al. 2019's argument, measured here).
- **The perception failure is real and is ours to fix**: {{PERCEPTION_VERDICT}}
- **The control failure is mostly the planner objective**, not memory: a 1.25 s CEM horizon with no value function,
  collision costing −5 against +10/m of progress, and reward heads read from embeddings that do not generalise.
- **V-JEPA 2-AC does not fit** (≈3–4 orders of magnitude over the CEM compute budget on a 6 GB laptop GPU, ViT-g
  + 300M predictor, RGB / Franka-specific, no memory beyond a 2–4 s window). At most a frozen feature extractor for
  a perception probe, where a depth-native ViT-S is better evidenced.
- **Minimal change to test the hypothesis**: (A) pixels-only encoder with state moved into the predictor's
  conditioning, then (B) a recurrent belief over the embeddings trained on long sequences with multi-step targets on
  data that actually revisits obstacles. See §6.

## 1. What we measured (recap)

| | DreamerV3 RSSM | LeWM ctx (3 frames) | LeWM emb | LeWM random init | frame CNN |
|---|---|---|---|---|---|
| probe AP, visible cells | **0.342** | 0.062 | 0.069 | 0.161 | 0.33 |
| probe AP, remembered (all ages) | **0.105** | 0.059 | 0.058 | 0.075 | 0.062 |
| probe AP, remembered, age 1–10 steps | **0.281** | 0.057 | 0.056 | 0.165 | 0.15 |
| depth decoder MAE train / held-out (m) | 0.045 / 0.058 | 0.16 / 0.53 | | | |

Base rates: visible 0.009, remembered 0.020. Dreamer's memory decays to frame level by ~40–80 steps (2–4 s).
LeWM CEM control (100 episodes per suite): goal 5 % / collision 82 % (random), 5 % / 86 % (U-trap), 0 % / 100 %
(start inside U or C trap; it flies ~1.2 m forward into the wall in ~1.3 s).

## 2. Literature

### 2.1 LeWM as published: what it was built and validated for

Maes, Le Lidec, Scieur, LeCun, Balestriero (2026), *LeWorldModel: Stable End-to-End JEPA from Pixels*,
arXiv:2603.19312; code github.com/lucas-maes/le-wm (checked at commit 8edfeb3).

- Encoder ViT-tiny (patch 14, 224² RGB); the embedding is the **[CLS] token only** + BatchNorm MLP projector
  (192-d). Patch tokens are never used.
- **Pixels-only**: proprioception is loaded by the data configs but never read by `JEPA.encode`; the paper calls
  the model pixels-only and removes proprio from the DINO-WM baseline "for a fair comparison".
- Loss: teacher-forced 1-step next-embedding MSE (history 3, frameskip 5) + 0.09·SIGReg.
- Planning: CEM on the latent distance to a **goal image**; no reward head; horizon 5 blocks.
- Environments: TwoRoom, PushT, OGBench-Cube, DMC Reacher, all fully observable from one frame (top-down or a
  fixed camera); evaluation start/goal pairs are drawn from the training set, and the train/val split is over
  windows, not layouts. **Nothing in the official protocol tests held-out layouts, partial observability or memory.**
- Probes in the paper are linear/MLP readouts of agent/object pose (r ≈ 0.97–0.99), with weak rotation and
  velocity (Cube joint velocity r = 0.095). No geometry/occupancy probe, no random-encoder baseline.
- Stated limitations: short-horizon planning; low data diversity "weakens SIGReg in simple, low-dimensional
  environments"; embedding dimension below ~184 hurts (we use 192).
- Zhang et al. (2026), *Frozen in a Frame*, arXiv:2610.04585: on the released LeWM weights, linear velocity probes
  are at or below chance while position probes reach R² ≈ 0.95: single-frame JEPA embeddings carry no motion.

### 2.2 What a JEPA / SIGReg latent keeps

- **SIGReg** (Balestriero & LeCun 2025, *LeJEPA*, arXiv:2511.08544) forces the batch marginal of the embedding to
  N(0, I). Its optimality results compare embeddings with the same information content; *which* input
  information is kept is decided entirely by the prediction task.
- **Slow / predictable features win.** Sobal et al. 2022 (arXiv:2211.10831): JEPAs "fail when the noise is fixed",
  as the loss "can be easily minimized by paying attention to noise that does not change with time". Littwin et
  al. 2024 (NeurIPS, arXiv:2407.03475): JEPAs are biased towards high-influence (highly predictive) features,
  increasingly with depth. Tang et al. 2023 (ICML, arXiv:2212.03319): self-predictive learning converges to the top
  eigenvectors of the transition operator, i.e. the most persistent modes.
- **LeWM-specific evidence (2026).** Liu et al., *Temporally Centered SIGReg*, arXiv:2607.26924: the raw LeWM
  objective "biases variance allocation toward the temporally persistent component", reducing decodability of
  robot dynamics; applying SIGReg to temporally centred residuals raises LIBERO success 63.6 → 83.8 %.
  Tan et al., *PokeWorld*, arXiv:2607.27017: a quantity becomes decodable when it is a prediction **target**;
  providing it only as **input** does not make it decodable, and in their RH20T table the trained input-only model
  reads it out worse than an untrained network (R² −0.16 vs 0.91), the same pattern as our trained-below-random
  probe. Zhu et al. 2026 (arXiv:2609.04264) name this "physical representation laziness": LeWM latents "do not
  collapse but nonetheless fail to represent key physical properties".
- **Global vs patch latents.** DINO-WM (Zhou, Pan, LeCun, Pinto, ICML 2025, arXiv:2411.04983), Table 2: DINOv2
  CLS vs patch features, success on Wall 0.58 vs 0.96, Reach 0.60 vs 0.92, PushT 0.44 vs 0.90; on unseen layouts
  (WallRandom) CLS 0.64 vs patch 0.82, DreamerV3 0.76. A single global vector loses "crucial spatial details".
- **Theory for histories.** Ni et al. 2024 (ICLR, arXiv:2401.08898): self-prediction (ZP) is weaker than
  observation prediction (belief), and needs reward prediction to pin task content; for POMDPs "start with
  recurrent networks as the encoder". LeWM's per-frame encoder does not satisfy their recurrence condition.

### 2.3 Memory in world models

- **Probing protocol.** Pasukonis, Lillicrap, Hafner 2022, *Memory Maze* (arXiv:2210.13383): frozen-state MLP
  probes (linear was too weak) for wall layout and object positions. RSSM with truncated BPTT (state carried
  across batches) beats plain RSSM, which "effectively resets the state to zero every 48 steps"; on large mazes
  models barely beat a no-memory baseline. Adding the probe target as an auxiliary loss helps the agent.
- **Sequence length bounds memory.** Samsami et al. 2024, *R2I* (ICLR, arXiv:2403.04253): DreamerV3 handles
  memory delays up to ~30 steps; models trained on 64-step sequences fail completely once the needed memory
  exceeds 64 steps; performance rises monotonically with training sequence length 64 → 1024. **This is very
  likely why our Dreamer probe decays by 40–80 steps (DreamerV3 batch length 64).**
- **Next-step prediction gives no pressure to remember.** Gregor et al. 2019, *Shaping belief states with
  generative environment models* (NeurIPS, arXiv:1906.09237): the immediate future is nearly a function of the last
  few observations, so ∇_b L ≈ 0 for the belief; longer overshoot (multi-step prediction, 4–32 steps) improves
  map decoding, and **deterministic predictors gain the least**. A deterministic L2 JEPA predictor is the worst case.
- **Data must contain revisits.** TECO (Yan et al., ICML 2023, arXiv:2210.02396) and long-context video WMs
  (S4WM, Deng et al. NeurIPS 2023, arXiv:2307.02064; Po et al. 2025, arXiv:2505.20171; WorldMem, arXiv:2504.12369)
  all build datasets where the camera returns to earlier content, because "many existing video benchmarks do not
  have strong long-range dependencies, where a model with limited context is sufficient".
- **Backbone choice matters less than training length.** S4WM's Memory Maze probes: RSSM-TBTT 95.0 % walls (9×9)
  vs TSSM-XL 86.4 vs S4WM 88.4. POPGym (Morad et al., ICLR 2023): "the GRU is the best general-purpose memory model".
- **JEPA predictors and object permanence.** V-JEPA shows object-permanence surprise (Garrido et al. 2025,
  arXiv:2502.11831) but the authors note "current JEPA models have limited memory" (3–4 s clips). Xie & Alanwar
  2026 (arXiv:2610.07355, preprint): the frozen V-JEPA 2 predictor loses a hidden moving object within 0.3 s even
  though the encoder still has it; 3k predictor-only steps on occlusion data install permanence: "the deficit is
  not intrinsic to latent prediction; what persists changes with the masks and the data."
- **Depth-sighted agents build no maps by themselves.** Wijmans et al. 2023 (ICLR, arXiv:2301.13261): blind LSTM
  agents' memories contain decodable occupancy (IoU 32.5 % vs 12.5 % untrained), but for depth-sighted agents
  "we do not find conclusive evidence that metric maps can be decoded": rich sensing admits reactive solutions.
  MAD (arXiv:2606.04534) reconstructs occupancy + visibility grids in a Dreamer RSSM for exactly this reason.

### 2.4 Trap escape: memory vs horizon / value

- AirDreamer (Liu et al. 2026, arXiv:2606.03252) attributes trap escape to the value function estimating "future
  returns beyond the rollout" and to emergent yaw scanning; it does not ablate memory.
- Meijer et al. 2024 (arXiv:2407.13530): an LSTM helps a learned reactive planner only where the current view is
  ambiguous; large or deep minima still fail.
- TD-MPC (Hansen et al., ICML 2022): finite-horizon MPC "yields only temporally local optimal solutions"; a
  terminal value fixes it. You et al. 2026 (arXiv:2607.10362): single-step latent error is "essentially
  uncorrelated with control success", since planners query off-manifold latents.

## 3. Verification: our implementation, or the nature of the task?

### 3.1 Code audit against the official repository

Matches: loss (`lejepa_forward`, teacher-forced over all causal positions), SIGReg (17 knots, 1024 projections,
λ = 0.09), ARPredictor (6 blocks, 16 heads, AdaLN-zero, dropout 0.1), BatchNorm MLP projector / pred_proj,
action Embedder over frameskip-concatenated actions, AdamW 5e-5, bf16, clip 1.0, batch 128, history 3, frameskip 5.

Differences (all deliberate, ordered by how much they could matter):

| ours | official | consequence |
|---|---|---|
| `emb = proj(CLS + Embedder(state))` | `emb = proj(CLS)` | state is part of the target and of the SIGReg budget: an easy, persistent, action-predictable component (§2.2) |
| state added after the ViT's final LayerNorm, unscaled | — | relative weight of state vs pixels unconstrained |
| 1-ch 48×80 depth, patch 8, 60 tokens | 224² RGB, patch 14, 256 tokens | CLS must summarise spatial occupancy of a cluttered egocentric view |
| reward / terminal heads, return-maximising CEM | goal-image latent distance | collision avoidance only through a post-hoc head |
| held-out layouts (episode-hash split) | in-distribution windows | generalisation never tested by the original paper |
| 30k steps ≈ 0.4M windows seen | 10 epochs ≈ 2–4M windows (paper) | possibly under-trained (secondary) |

### 3.2 Experiment 1: how much memory does the objective ask for? (data only)

`scripts/diag/reobs.py`: for obstacle cells visible at t+k, where was that information available to LeWM's
predictor at time t (frames t−10, t−5, t)?

| target | in the 3 context frames | never seen (novel) | seen 1–15 steps ago, not in a context frame | 16–64 steps ago | > 64 steps ago |
|---|---|---|---|---|---|
| next frame, k = +5 (all episodes) | 0.909 | 0.067 | 0.014 | 0.005 | 0.005 |
| next frame, k = +5 (trap episodes) | 0.921 | 0.056 | 0.013 | 0.004 | 0.005 |
| CEM horizon end, k = +25 (all) | 0.668 | 0.287 | 0.016 | 0.011 | 0.017 |
| CEM horizon end, k = +25 (trap) | 0.717 | 0.241 | 0.015 | 0.009 | 0.017 |

Memory beyond the LeWM window could explain at most ~1 % of the next-frame obstacle content (2.6–2.8 % at the
1.25 s horizon). Even a perfect memory would barely lower LeWM's training loss, so **this data + objective cannot
elicit memory, in LeWM or any 1-step predictor**: a property of the task setup, not of our code. The same holds
for Dreamer's 1-step objective, which is consistent with its memory being short (it is carried passively by the
GRU, then fades).

### 3.3 Experiment 2: where is the obstacle geometry lost?

{{DIAG_DECODE}}

### 3.4 Experiment 3: what does the trained predictor use?

{{DIAG_PREDICT}}

### 3.5 Verdict

{{VERDICT}}

## 4. Where exactly LeWM fails, layer by layer

{{LAYERS}}

## 5. Would V-JEPA 2-AC help?

Facts (Assran et al. 2025, arXiv:2506.09985; repo facebookresearch/vjepa2; V-JEPA 2.1, arXiv:2603.14482):

- V-JEPA 2 encoders: ViT-L 300M, ViT-H 600M, ViT-g 1B (256 px, 2×16×16 tubelets); **no small V-JEPA 2**.
  V-JEPA 2.1 (Mar 2026) adds distilled ViT-B 80M / ViT-L 300M at 384 px with dense features.
- V-JEPA 2-AC: frozen ViT-g encoder, ~300M block-causal predictor, 256² RGB at 4 fps, 7-D end-effector actions
  and state, < 62 h of DROID, trained on 32 GPUs. CEM 800 samples × 10 iterations × **horizon 1** against a goal
  image: **16 s per action on an RTX 4090**. Checkpoint 11.8 GB (CPU RAM is 14 GB). JEPA-WMs (Terver et al., TMLR
  2026, arXiv:2512.24497) report a bug in the released rollout loss.
- Memory: none beyond attention over its 8–16-frame context, and the released planner uses a **single context
  frame**. Its latent is spatial (patch tokens), but Xie & Alanwar 2026 show its predictor drops hidden objects
  within ~0.3 s.

Estimates for this laptop (≈10 TFLOP/s effective fp16, unmeasured): one 800×10×1 plan ≈ 1.3 PFLOP ≈ 2 min per
action for one drone; 16 drones at 4 Hz need ~8000× the budget. The budget per plan (≈150 GFLOP) is about one
forward of the AC predictor; even DINO-WM-sized patch predictors (19M params, 196 tokens) do not fit. A global or
few-token latent like LeWM's is the regime that fits the fleet CEM.

Verdict: **no as a world model.** It would not test the memory hypothesis either (no persistent state, 1-frame
planning context), it needs retraining for 5-D velocity actions and depth input that this GPU cannot do, and there
is no evidence of V-JEPA transfer to depth or drones. The only sensible use is offline: a frozen V-JEPA 2.1 ViT-B
(or better evidenced on depth, DeFM-S, arXiv:2601.18923, or DINOv2-S) as a *perception* baseline for the visible-
cell probe. That answers "is a pretrained dense encoder better at seeing obstacles", which is not our question.

## 6. Minimal architecture changes to test the hypothesis

Hypothesis: *a JEPA latent world model can keep out-of-view obstacles if (i) its latent encodes the geometry,
(ii) it has somewhere to keep the past, and (iii) its objective rewards keeping it.* LeWM currently fails all three.

{{PLAN}}

## References

LeWM 2603.19312 · le-wm code (8edfeb3) · LeJEPA 2511.08544 · Sobal et al. 2211.10831 · Littwin et al.
2407.03475 · Tang et al. 2212.03319 · Ni et al. 2401.08898 · TC-SIGReg 2607.26924 · PokeWorld 2607.27017 ·
Spectral-target 2609.04264 · Frozen in a Frame 2610.04585 · DINO-WM 2411.04983 · PLDM 2502.14819 · JEPA-WMs
2512.24497 · TD-MPC 2203.04955 · TD-MPC2 2310.16828 · You et al. 2607.10362 · Memory Maze 2210.13383 · R2I
2403.04253 · Gregor et al. 1906.09237 · S4WM 2307.02064 · TECO 2210.02396 · Po et al. 2505.20171 · WorldMem
2504.12369 · POPGym 2303.01859 · Garrido et al. 2502.11831 · Xie & Alanwar 2610.07355 · Wijmans et al. 2301.13261 ·
MAD 2606.04534 · AirDreamer 2606.03252 · Meijer et al. 2407.13530 · V-JEPA 2 2506.09985 · V-JEPA 2.1 2603.14482 ·
DeFM 2601.18923 · DreamerV3 2301.04104 (Nature 2025).
