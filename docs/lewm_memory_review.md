# Memory for LeWM: diagnosis, literature review and extension plan

Date: 2026-10-21. Data: `data/kin_v1` (1000 kinematic stand-in episodes, scripted goal / trap / explore policy).
Runs: `runs/kin_v1/{lewm,wm,probe,probe_untrained,rl}`. Diagnostics: `scripts/diag/`, results in
`runs/kin_v1/lewm/diag.json`.

**Scope.** This document is about the reconstruction-free, latent-predictive path (LeWM / JEPA). The DreamerV3
(RSSM, reconstruction) memory line, with its probe, map auxiliary loss and online RL, is colleagues' work. Here it
appears only as a reference column measured on the same data.

## 0. Summary

1. **Diagnosis.** LeWM's code matches the official repository. On this task it fails at three points:
   - **The global read-out.** The ViT's patch tokens hold the geometry: held-out depth to 2.7 cm. The single CLS
     vector keeps only coarse geometry (0.32 m).
   - **Our state fusion.** Summing the body state into the embedding makes every read-out memorise training
     layouts: 0.09 m train vs 0.62 m held-out.
   - **The architecture and objective.** Memory is impossible by construction: a per-frame latent and a 0.75 s
     context. It is also unrewarded by the objective: only 1 % of next-frame obstacle content needs memory beyond
     the window.
2. **The literature says memory in a predictive latent is possible but must be built and demanded.** Every
   reconstruction-free model with measured memory has the same three parts:
   - a closed-loop recurrent belief over single-frame embeddings;
   - multi-step open-loop prediction of future frame embeddings from that belief;
   - data in which looked-away content returns.

   The models are CPC|Action, PBL, BYOL-Explore, NE-Dreamer, and VAE+GRU in Memory Maze. 1-step prediction gives
   ~0 gradient to the belief (Gregor et al. 2019), and the V-JEPA 2 predictor drops hidden objects within 0.3 s
   unless trained on occlusion data (Xie & Alanwar 2026).
3. **The gap.** No 2025–26 JEPA world model tests memory of out-of-view content under egocentric partial
   observability:
   - The JEPA navigation models (InJePA, Skytopia, NWM) use short frame windows and report task success only.
   - The closest prior work evaluates only on the fully observable LeWM benchmarks. That is a JEPA world model
     with an associative memory bank (Sun & Zhu 2026, arXiv:2610.19683).
   - Nobody regularises a recurrent JEPA belief with SIGReg.
4. **Plan.** R-LeWM is a short ladder of minimal changes:
   - pixels-only encoder (state into the predictor's conditioning);
   - a few spatial tokens;
   - a GRU belief with a multi-step open-loop head and truncated BPTT;
   - training data with memory pressure.

   Evaluation is label-free first: re-entry prediction error, surprise on layout edits, lesions and transplants of
   the belief, and closed-loop trap escape. Occupancy probes come last. The component, bandwidth and frame
   analyses are chosen so that their answers carry over to multi-agent world-model design.
5. **V-JEPA 2-AC does not fit.** It is 3–4 orders of magnitude over the fleet CEM budget, has no persistent
   memory, and is RGB- and Franka-specific. At most it could serve as a frozen perception baseline.

## 1. Measurements on `kin_v1`

| | LeWM ctx (3 frames) | LeWM emb | LeWM random init | frame CNN | *reference: DreamerV3 RSSM* |
|---|---|---|---|---|---|
| probe AP, visible cells (base 0.009) | 0.062 | 0.069 | 0.161 | 0.33 | *0.342* |
| probe AP, remembered cells (base 0.020) | 0.059 | 0.058 | 0.075 | 0.062 | *0.105* |
| probe AP, remembered, ≤ 0.5 s | 0.057 | 0.056 | 0.165 | 0.15 | *0.281* |
| depth decoder, train / held-out MAE (m) | | 0.09 / 0.62 | 0.11 / 0.18 | | *0.045 / 0.058* |

Closed loop, 100 episodes per suite, goal / collision:

| suite | LeWM + CEM | *reference: DreamerV3 online RL (5e5 steps)* |
|---|---|---|
| random | 5 % / 82 % | *98 % / 2 %* |
| U-trap (approach) | 5 % / 86 % | *98 % / 2 %* |
| start inside U-trap | 0 % / 100 % | *65 % / 11 %* |
| C-trap | 0 % / 100 % | *16 % / 44 %* |

The CEM comparison is not like-for-like: a 1.25 s planning horizon with no value function, against a learned
actor-critic. The trap rows are the target for a memory-equipped LeWM.

## 2. Diagnosis of LeWM on this task

### 2.1 Code audit against the official repository (lucas-maes/le-wm @ 8edfeb3)

**Matches the official code:**
- the loss (`lejepa_forward`, teacher-forced over all causal positions);
- SIGReg (17 knots, 1024 projections, λ = 0.09);
- ARPredictor (6 blocks, 16 heads, AdaLN-zero, dropout 0.1);
- the BatchNorm MLP projector;
- the action Embedder over frameskip-concatenated actions;
- the optimiser (AdamW 5e-5, bf16, clip 1.0), batch 128, history 3, frameskip 5.

The official `JEPA.encode` reads only pixels and actions. Proprioception is loaded by the data configs but never
used.

| ours | official | consequence |
|---|---|---|
| `emb = proj(CLS + Embedder(state))` | `emb = proj(CLS)` | state becomes part of the target and of the SIGReg budget (§2.3–2.4) |
| 1-ch 48×80 depth, patch 8, 60 tokens | 224² RGB, patch 14, 256 tokens | one CLS vector must summarise an egocentric cluttered view |
| reward / terminal heads, return-maximising CEM | latent distance to a goal image | collision avoidance only through a post-hoc head |
| held-out layouts (episode-hash split) | train/val split over windows; goals drawn from the training set | generalisation never tested by the original paper |
| 30k steps ≈ 0.4M windows | 10 epochs ≈ 2–4M windows | possibly under-trained (secondary) |

### 2.2 How much memory does the objective ask for? (`scripts/diag/reobs.py`, data only)

Take the obstacle cells visible at t+k. Was their information available to LeWM's predictor at time t, which
sees frames t−10, t−5 and t?

| target | in the 3 context frames | never seen | last seen 1–15 steps ago | 16–64 | > 64 |
|---|---|---|---|---|---|
| next frame, k = +5 (all episodes) | 0.909 | 0.067 | 0.014 | 0.005 | 0.005 |
| next frame, k = +5 (trap episodes) | 0.921 | 0.056 | 0.013 | 0.004 | 0.005 |
| CEM horizon end, k = +25 (all) | 0.668 | 0.287 | 0.016 | 0.011 | 0.017 |
| CEM horizon end, k = +25 (trap) | 0.717 | 0.241 | 0.015 | 0.009 | 0.017 |

Even a perfect memory would lower LeWM's training loss on at most ~1 % of next-frame obstacle content, and
~2.8 % at a 1.25 s horizon. **This data and this objective cannot produce memory.** This is a property of the
setup, not a code bug, and it sets two of the design constraints in §5: multi-step targets and revisit-rich data.

### 2.3 Where is the obstacle geometry lost? (`scripts/diag/lewm_diag.py`, part A)

Setup:
- The same depth decoder for each read-out (a matching conv decoder for the 6×10 patch grid).
- 3000 steps on 120 train episodes; scored on 60 episodes with held-out layouts.
- Baseline: a constant mean image scores 1.03 m.
- "near" = pixels closer than 2.5 m.

| read-out (192-d unless noted) | train MAE (m) | held-out MAE (m) | held-out near (m) |
|---|---|---|---|
| trained: 60 patch tokens (60×192) | 0.025 | **0.027** | 0.046 |
| random init: 60 patch tokens | 0.017 | 0.018 | 0.023 |
| trained: CLS, before the state is added | 0.263 | **0.315** | 0.744 |
| random init: final emb | 0.105 | 0.178 | 0.339 |
| trained: final emb = proj(CLS + state) | 0.091 | **0.617** | 1.641 |
| trained: state embedding alone | 0.363 | 0.776 | 2.827 |

- **The encoder is fine.** Its patch tokens reconstruct held-out depth to 2.7 cm, with no generalisation gap.
- **The CLS read-out keeps only coarse geometry.** Training makes the global vector *worse* than random weights,
  in both the decoder (0.62 vs 0.18 m) and the probe (AP 0.06 vs 0.16). The objective discards pixel detail from
  the global embedding.
- **The state is the layout-memorisation shortcut.** The state embedding alone decodes *training* depth well
  (0.36 m) because it identifies the scripted start/goal configuration, but it generalises worst.
- The decoder, the probe and the reward head all read the final embedding, so all three inherit this shortcut.

### 2.4 What does the trained predictor use? (part B, 4361 held-out windows)

Each row replaces one part of the context with its dataset mean; the target is unchanged.

| context | next-embedding MSE |
|---|---|
| full context | 0.008 |
| copy last frame | 0.148 |
| actions shuffled | 0.450 |
| state part → mean (pixels kept) | 0.584 |
| pixel part → mean (state kept) | 0.599 |
| target variance | 1.019 |
| *target moves, if its pixel part → mean* | 0.594 |
| *target moves, if its state part → mean* | 0.830 |

- Prediction is near-perfect in distribution and uses pixels, state and actions, so it is not a pure shortcut.
- The target, however, is state-dominated: under SIGReg most of the 192 unit-variance directions go to the body
  state and to slow, coarse view features.

### 2.5 Verdict

| failure | cause | whose |
|---|---|---|
| probe below random init; decoder 7× held-out gap | state fused into the target embedding | **ours**: deviation from pixels-only LeWM |
| global latent keeps only coarse geometry | one CLS vector, plus the JEPA bias toward persistent, predictable features | **LeWM design** vs an egocentric cluttered view |
| no out-of-view memory | per-frame latent, 0.75 s context, ~1 % memory demand in a 1-step target | **design + task**: untestable in LeWM as specified |
| CEM collides 82–100 % | 1.25 s horizon, no value, cheap collision term, non-generalising reward head | **our planner adaptation** |

## 3. Literature review

### 3.1 LeWM, and what a JEPA / SIGReg latent keeps

- **LeWM** (Maes, Le Lidec, Scieur, LeCun, Balestriero 2026, arXiv:2603.19312).
  - Design: CLS-only, pixels-only, 1-step teacher-forced prediction plus SIGReg, CEM to a goal image.
  - Evaluation: fully observable tasks only (TwoRoom, PushT, OGBench-Cube, Reacher), with no held-out layouts.
  - Probes: pose decodes well (r ≈ 0.97–0.99), but rotation and velocity decode poorly (Cube joint velocity
    r = 0.095).
  - Surprise tests: teleport surprise is significant; colour change is not.
  - Stated limitations: short horizons and low data diversity. Memory is not discussed.
  - Zhang et al. 2026, *Frozen in a Frame* (arXiv:2610.04585): velocity probes on the released LeWM weights are at
    or below chance.
- **SIGReg** (LeJEPA, arXiv:2511.08544) fixes the batch marginal to N(0, I). *Which* information is kept is set by
  the prediction task.
- **Slow and predictable features win.** Sobal et al. 2022 (arXiv:2211.10831); Littwin et al. 2024 (NeurIPS,
  arXiv:2407.03475; high-influence features); Tang et al. 2023 (ICML; the most persistent eigenmodes).
  - H-JEPA (arXiv:2610.06805, App. L) defines "slow-feature collapse": an encoder keeps the episode-constant
    background and drops the moving agent. This is "always feasible when the regularizer sees only the marginal
    law".
  - Liu et al. 2026 (arXiv:2607.26924): raw LeWM "biases variance allocation toward the temporally persistent
    component". Their fix applies SIGReg to temporally centred residuals.
- **Input is not target.** PokeWorld (Tan et al. 2026, arXiv:2607.27017): a quantity becomes decodable when it is
  predicted, not when it is only given as input. In their RH20T table the trained input-only model reads it out
  below an untrained network. This is the pattern of our state fusion.
- **Global vs spatial latents.** DINO-WM (ICML 2025, arXiv:2411.04983), CLS vs patch success: Wall 0.58 vs 0.96,
  PushT 0.44 vs 0.90, unseen layouts 0.64 vs 0.82.

### 3.2 How memory gets into reconstruction-free predictive models

| model | belief (closed loop) | prediction (open loop) | target | anti-collapse | memory evidence |
|---|---|---|---|---|---|
| CPC\|Action (Guo et al. 2018, arXiv:1811.06407) | GRU over frame embeddings + actions | action-GRU, f ∈ [1, 30] | future frame embedding | contrastive | position loss on DMLab `terrain`: 30-step 1.589 vs 1-step 3.348; objects remembered after looking away **only when they later change observations** |
| PBL (Guo et al. 2020, arXiv:2004.14646) | agent LSTM | 20 steps | future frame latent, **plus reverse: frame latent → belief** | stop-grad both ways | DMLab-30 gains; horizon 1 = no gain; red-cube position "much better remembered after the agent has looked away" |
| BYOL-Explore (Guo et al. 2022, arXiv:2206.08332) | GRU | K = 8–10 | EMA frame embedding | EMA + stop-grad | DM-HARD-8: K=8 69.9 vs K=1 50.6 (partially observable only); fixed targets 19.9; longer training sequences (64→128) help |
| NE-Dreamer (Bredis et al. 2026, arXiv:2603.02765) | RSSM + causal transformer | next step | **next** frame embedding | Barlow Twins | beats DreamerV3 on DMLab memory tasks; matching the *current* rather than the next embedding loses nearly all gains |
| VAE+GRU (Memory Maze, arXiv:2210.13383) | GRU, truncated BPTT | next step | frozen VAE embedding (MSE) | frozen target | walls 90.0 % vs no-memory 84.1 % and RSSM-TBTT 94.0 % (9×9) |
| ZP (Ni et al. 2024, ICLR, arXiv:2401.08898) | LSTM | next step | next belief (EMA) | EMA | MiniGrid memory tasks: ZP > model-free, but < observation prediction; EMA keeps belief rank, online/detached targets lose it |

Mechanisms:
- **Gregor et al. 2019** (arXiv:1906.09237): next-step targets give ∇_b L ≈ 0. Longer overshoot improves map and
  position decoding, and deterministic predictors gain least.
- **R2I** (ICLR 2024): a non-recurrent posterior q(z_t | o_t) "retains or elevates" memory performance, so memory
  can live entirely in the sequence model. That is exactly the split of a single-frame LeWM encoder plus a
  recurrent belief.
- **Xie & Alanwar 2026** (arXiv:2610.07355): the V-JEPA 2 predictor loses a hidden moving object within 0.3 s
  while its encoder still holds it. 3k predictor-only steps on occlusion data install it (belief 0.05 → 1.00):
  "the deficit is not intrinsic to latent prediction; what persists changes with the masks and the data."
- **JEPA-WMs** (Terver et al., TMLR 2026, arXiv:2512.24497): a 2-step rollout loss is best for planning in
  simulation and 6 steps on DROID, and too-long training context hurts. Keep the planning head short-horizon even
  if the belief is trained long.
- **UWM-JEPA** (arXiv:2605.25313, toy): with teacher-forced belief targets the action pathway becomes inert
  (‖H1‖/‖H0‖ ≈ 0.03). Test action sensitivity explicitly.

### 3.3 Recurrent / memory JEPAs, 2025–26, and the gap

| work | memory | tasks / evaluation |
|---|---|---|
| Sun & Zhu 2026, *Enhancing Latent World Models with Associative Memory* (arXiv:2610.19683) | content-addressable slot memory on JEPA prediction, multi-timescale | PushT, TwoRoom, Reacher, Cube: fully observable, long-horizon planning |
| H-JEPA (arXiv:2610.06805) | hierarchy, no recurrence | AntMaze 18 → 73 % |
| seq-JEPA (NeurIPS 2025, arXiv:2505.03176) | transformer over (view, action) | path integration over viewpoints; no control |
| UWM-JEPA (arXiv:2605.25313) | density-matrix belief | toy hidden-velocity task |
| EpicWorldModel (arXiv:2610.05996) | H-frame window, stochastic JEPA | success stratified by goal visibility: occluded goal 79.5 vs LeWM 33.3 |
| InJePA (arXiv:2610.23896), Skytopia (arXiv:2609.26007), NWM (CVPR 2025) | 4–8-frame windows; Skytopia "lacks persistent spatial memory" | navigation success only |

No work tests whether a JEPA world model's latent remembers content that left an egocentric view, and none
regularises a recurrent belief with SIGReg. That is the gap.

### 3.4 Spatial and ego-motion-structured memory

- **GTM-SM** (Fraccaro et al., ICML 2018, arXiv:1804.09401) factorises a learned position state (integrated from
  actions) from a frame code, with memory retrieved by position. It predicts coherently > 200 steps after
  memorisation. This is the closest ancestor of our percept / ego-motion / belief split.
- **TEM** (Whittington et al., Cell 2020): a path-integrated structural code plus a separate sensory code, bound in
  memory and trained by next-observation prediction only.
- **Flow Equivariant World Models** (Lillemark et al., ICML 2026, arXiv:2601.01075): a latent map larger than the
  field of view, transported by the inverse self-motion. Block positions are decodable at 96 % (< 1 % for
  baselines), and its equivariance error is a reusable label-free diagnostic. EgoMap (arXiv:2002.02286) stores
  maps allocentrically to avoid resampling blur and tolerates ~10 % odometry noise.
- **Conditions for ego-motion integration to emerge:**
  - the objective requires it (multi-step prediction, positional targets);
  - action input reaches the recurrent state;
  - the data contains revisits;
  - a capacity or noise bottleneck.

  Evidence: Gregor; Levenstein et al. 2024; Cueva & Wei, ICLR 2018; Schaeffer et al., NeurIPS 2022 (integration
  emerges easily when it is the target; grid codes do not, so test function, not tuning curves). Gornet & Thomson
  (Nat. Mach. Intell. 2024): next-image prediction builds better spatial maps than an autoencoder on the same
  data.
- **Rich sensing allows reactive solutions without maps.** Wijmans et al. 2023 (ICLR) find occupancy in blind
  agents' memories (IoU 32.5 % vs 12.5 % untrained), but nothing conclusive for depth-sighted agents.

### 3.5 Evaluating memory without labels

- **Re-entry / revisit consistency.**
  - TECO (ICML 2023) builds data where content leaves view and returns; FVD saturates while long-range
    consistency still improves.
  - Po et al. 2025 (arXiv:2505.20171), backtrack retrieval: replay the inverse action sequence, and the ground
    truth is the reversed past.
  - WorldMem (arXiv:2504.12369): pose-indexed retrieval matters.
- **Surprise / violation of expectation.** Garrido et al. 2025 (arXiv:2502.11831) define surprise as the latent
  prediction error in a paired possible/impossible protocol, with an untrained-network control. LeWM's own
  teleport test uses the same idea.
- **Causal lesions and transplants.**
  - Wijmans memory budget: truncate to the last k steps.
  - Wijmans probe agent: an agent initialised with a predecessor's memory takes shortcuts, SPL 85.0 vs 62.9.
  - Interchange interventions (Geiger et al., NeurIPS 2021).
  - POPGym Arcade (arXiv:2503.01450): out-of-distribution inputs contaminate memory "well into the future".
  - Shchendrigin et al. 2026 (arXiv:2601.15086): rewriting memory must be tested separately from retaining it.
- **Probe after, not instead.** Probe decodability is not use (Xie & Alanwar: present in the encoder, absent from
  the predictor). Memory Maze protocol: frozen states, MLP probe, scored after burn-in.

### 3.6 Trap escape: memory vs planning horizon

- AirDreamer (arXiv:2606.03252) credits the value function, "future returns beyond the rollout", and emergent yaw
  scanning. It does not ablate memory.
- TD-MPC (ICML 2022): finite-horizon MPC gives only locally optimal solutions, and a terminal value fixes it.
- You et al. 2026 (arXiv:2607.10362): single-step latent error is "essentially uncorrelated with control success".
- For LeWM the planner must therefore be fixed (value or longer horizon) *before* closed-loop trap success can
  measure memory.

## 4. V-JEPA 2-AC (verdict: no)

V-JEPA 2-AC (arXiv:2506.09985) is a frozen ViT-g encoder (1B) with a ~300M block-causal predictor. It works on 256²
RGB at 4 fps with 7-D end-effector actions, plans one step with CEM (800 × 10) to a goal image, and takes 16 s per
action on an RTX 4090. Its checkpoint is 11.8 GB, against 14 GB of RAM here. There is no memory beyond an 8–16-frame
context, and the released planner uses one context frame. V-JEPA 2.1 (arXiv:2603.14482) adds a distilled ViT-B
(80M).

On this laptop (≈10 TFLOP/s, an estimate), one plan takes ≈ 2 min per drone; 16 drones at 4 Hz need ~8000× the
budget. It would also not test memory, it needs retraining for 5-D velocity actions and depth input, and there is
no evidence that V-JEPA transfers to depth.

The only use is an offline frozen perception baseline. For depth, DeFM-S (arXiv:2601.18923) or DINOv2-S is better
evidenced. Section 2.3 already shows our own encoder's patch tokens hold the geometry, so a bigger encoder is not
what is missing.

## 5. Extension plan: R-LeWM

Hypothesis: a reconstruction-free predictive latent keeps out-of-view obstacles if three conditions hold:
- (i) its latent encodes the geometry;
- (ii) it has a persistent state;
- (iii) its objective and data make remembering pay off.

LeWM currently fails all three. Each step below is one minimal change, evaluated against the previous step.

### 5.1 Model ladder

| step | change (code size) | addresses | expected effect |
|---|---|---|---|
| **M0 pixels-only** | `emb = proj(CLS)`; state concatenated to the action block in the predictor's AdaLN conditioning and given to the reward head (~10 lines, official-like) | §2.3 shortcut | held-out decoder/probe close to the random-init level or better; the train/held-out gap closes |
| **M1 spatial tokens** | predictor over the CLS + a pooled patch grid (e.g. 3×5 = 16 tokens, 192-d); SIGReg per token | §2.3 coarse CLS; DINO-WM | visible-cell AP toward the frame CNN (≈ 0.33); still no memory |
| **M2 recurrent belief** | GRU b_t = f(b_{t-1}, e_t, a_{t-1}), the predictor conditioned on b_t; an open-loop action-only head predicting e_{t+k}, k = 1…K (K ≈ 8–16 blocks = 2–4 s); targets = future frame embeddings; truncated BPTT on 32–64-frame sequences | §2.2, §3.2 | memory appears in b_t; re-entry error drops vs M1 for gaps up to ~K |
| **M3 memory pressure** | look-away/look-back collection (yaw scans, trap exits) so that the reobs memory fraction ≫ 1 %; blind steps (observation dropout); up-weight re-entry frames | §2.2, Xie & Alanwar | memory horizon extends; tested by the M1/M2 metrics below |
| M4 (optional) | ego-motion-equivariant belief (latent map transported by the known body motion, FloWM-style) | §3.4 | memory that survives yaw; a frame-aware belief |

Design choices fixed by the literature:
- Keep SIGReg and the BatchNorm projector on frame embeddings. MuDreamer finds BatchNorm crucial against collapse.
- Monitor the belief's rank and within-episode variance; the belief is otherwise unregularised. SIGReg on the
  belief is an experiment, not a recipe.
- Ablate EMA / stop-gradient on the multi-step targets (Ni, BYOL-Explore).
- Use a strict next-step shift (NE-Dreamer).
- Optionally add PBL-style reverse prediction (frame embedding → belief).
- Keep a short-horizon head for CEM (JEPA-WMs).

The planner fix is orthogonal to memory but needed before closed-loop numbers mean anything: a value head on b_t
(TD on embeddings, TD-MPC style) or a collision cost. Cost estimate: ≈ 3–5 h per 15k-step run on this laptop,
extrapolated from the 30k-step LeWM run (5 h).

### 5.2 Evaluation protocol (label-free first)

| id | measure | what it shows |
|---|---|---|
| P0 | scripted look-away/look-back passes with gap Δ ∈ {5, 10, 20, 40, 80, 160} steps, each with a **novel twin** (same geometry never seen before) | controls: published LeWM, untrained network, M1 (no belief) |
| E1 | **re-entry contrast**: latent prediction error on re-entering frames, novel − revisit, minus the same for the memoryless baseline, against Δ; plus backtrack retrieval | retention curve and half-life, without labels |
| E2 | **surprise on layout edits** while the obstacle is out of view (removed, moved, added), paired protocol | the belief encodes *this* layout, not a template |
| E3 | **lesions**: reset b to its init k steps before re-entry; shuffle b across layouts; ablate ego-motion during the gap; blind the percept during the gap; transplant b from another pass through the same layout | causality, effective horizon, path integration |
| E4 | **closed loop**: CEM trap escape with intact / reset / transplanted belief, stratified by initial obstacle visibility | memory used for control |
| E5 | occupancy probe (existing `probe/probe.py`) on b, with untrained and memoryless controls | corroboration; comparable to the Dreamer reference |

Statistics: ≥ 3 seeds, paired tests, bootstrap confidence intervals. Pitfall: re-entry frames have higher error by
nature, so the novel twin and the memoryless baseline are mandatory.

### 5.3 Component analyses that carry over to multi-agent world models

| analysis | single-agent result | multi-agent design decision |
|---|---|---|
| which component holds memory (percept e, ego-motion path, belief b), via E3 per component | where out-of-view information lives | communicate observation embeddings (MA-JEPA, arXiv:2609.33563) or beliefs (MAMBA, arXiv:2205.15023)? No paper compares the two for JEPAs |
| rate curve: Gaussian-noise / quantisation bottleneck on b, E1/E4 against bits (½·log₂(1 + SNR) per dimension; cross-checked with CCA / KSG) | how many bits obstacle memory needs | message budget (MAMBA sends 160 bits; V2VNet 417× compression; IMAC) |
| reference frame: equivariance error of b under known ego-motion; transplant with a pose offset, with and without a transform; pose noise 0–20 % | is the belief egocentric? | must messages carry a pose and be warped (V2VNet), or can they be fused directly? |
| half-life (E1) and contamination (transplant a conflicting belief, measure the overwrite time) | how stale a belief may be; how fast observations correct it | message freshness and trust |
| belief fusion: A saw the left wall, B saw the right; average vs pseudo-observation injection vs learned fusion, scored by E1/E2 on both walls | whether beliefs compose | the fusion operator; no study exists for JEPA beliefs |

## 6. Risks and open questions

- **Untrained belief targets.** Multi-step targets from the same online encoder, with no EMA, could co-adapt
  (JEPA-x, arXiv:2608.24044). Ablate EMA targets.
- **Slow-feature collapse of the belief** into an episode constant (H-JEPA App. L). Monitor the within-episode
  variance of b.
- **Memory pressure from data may still be small** for a scripted pilot. Check the reobs fraction on the new data
  before training (target ≥ 10 % of re-entry content).
- **CEM vs RL.** The closed-loop comparison with the Dreamer reference stays confounded by the planner (§3.6). Keep
  claims about memory to E1–E3, and use E4 only within LeWM variants.
- **Stand-in simulator.** All of this is on the kinematic stand-in (analytic depth). Isaac Sim is blocked by
  driver 595 on this machine.

## References

LeWM 2603.19312 · le-wm code (8edfeb3) · LeJEPA 2511.08544 · Frozen in a Frame 2610.04585 · Sobal et al. 2211.10831 ·
Littwin et al. 2407.03475 · Tang et al. 2212.03319 · H-JEPA 2610.06805 · TC-SIGReg 2607.26924 · PokeWorld 2607.27017 ·
JEPA-x 2608.24044 · DINO-WM 2411.04983 · CPC|Action 1811.06407 · PBL 2004.14646 · BYOL-Explore 2206.08332 ·
NE-Dreamer 2603.02765 · Memory Maze 2210.13383 · Ni et al. 2401.08898 · Gregor et al. 1906.09237 · R2I 2403.04253 ·
Xie & Alanwar 2610.07355 · JEPA-WMs 2512.24497 · UWM-JEPA 2605.25313 · MuDreamer 2405.15083 · Sun & Zhu 2610.19683 ·
seq-JEPA 2505.03176 · EpicWorldModel 2610.05996 · InJePA 2610.23896 · Skytopia 2609.26007 · NWM 2412.03572 ·
GTM-SM 1804.09401 · TEM (Cell 2020) · FloWM 2601.01075 · EgoMap 2002.02286 · Neural Map 1702.08360 · Cueva & Wei
1803.07770 · Schaeffer et al. (NeurIPS 2022) · Gornet & Thomson 2308.10913 · Wijmans et al. 2301.13261 · TECO
2210.02396 · Po et al. 2505.20171 · WorldMem 2504.12369 · Garrido et al. 2502.11831 · Geiger et al. 2106.02997 ·
POPGym Arcade 2503.01450 · Shchendrigin et al. 2601.15086 · AirDreamer 2606.03252 · TD-MPC 2203.04955 · You et al.
2607.10362 · V-JEPA 2 2506.09985 · V-JEPA 2.1 2603.14482 · DeFM 2601.18923 · MA-JEPA 2609.33563 · MAMBA 2205.15023 ·
V2VNet 2008.07519 · IMAC 1911.06992 · DreamerV3 2301.04104.
