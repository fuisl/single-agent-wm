# Single-agent memory in a latent-predictive world model (R-LeWM)

Status: 2026-10-21 22:35. M0 and M1 done, M2 training (≈ 2.7 h per run, eval included ≈ 3.2 h), M3–M4 queued
(chain in tmux `lewm_gpu`, log `logs/lewm_chain.log`).
Companion documents:
- `docs/lewm_memory_review.md`: the diagnosis, the literature review and the reasons for each design choice.
- `docs/uav_memory_probe_kit.md`: the occupancy probe and the data format.

This document is the self-contained account of the memory question itself:
- why it matters for this task;
- what we mean by memory and how we measure it;
- what we built to get it into LeWM;
- what we have measured so far and what would count as success.

All results are on the kinematic stand-in simulator (`SIM=kin`, analytic depth; Isaac Sim is blocked by driver
595). DreamerV3 numbers are a reference column only: that line is colleagues' work.

---

## 1. Why a world model for this drone needs memory

**The task is partially observable.** The drone flies through clutter with a 1-channel 48×80 depth camera:
- 87° horizontal field of view;
- 0.6–6 m range;
- the drone yaws, glances away, backs out of traps and turns around.

Most of the obstacles that constrain the next few seconds of flight are, at any moment, **outside the image**.

**A planner is only as good as the futures it imagines.** LeWM plans with CEM: it rolls candidate action
sequences through the predictor and scores the predicted latents. Suppose the latent at time t carries only what
the last few frames show. Then every imagined future in which the drone turns towards a wall it saw five seconds
ago is predicted as free space. The planner will choose those futures.

**The decision-theoretic object is a belief state.** In a POMDP the sufficient statistic for acting is
b_t = p(world | o_≤t, a_<t), the posterior over the world given everything observed so far.
- A window model approximates it by p(world | o_{t−2..t}). That is exact only if three frames determine
  everything that matters.
- For a moving narrow-FoV camera that is false in exactly the situations that matter: turning back, escaping a
  U/C trap, re-approaching an obstacle after a glance.

**Memory is therefore not the goal in itself.** It is the property that lets the latent serve as a planning
state when decisions depend on things out of view. The closed-loop failures we already see are in the traps,
where the exit is behind or beside the drone:
- start inside a U-trap: LeWM + CEM 0 % success;
- C-trap: LeWM + CEM 0 % success, against 16–65 % for the Dreamer agent.

Those are the cases where a memoryless latent is at its weakest.

## 2. What "memory" means here

**Definition.** A model has memory to the extent that its state at time t carries information about the world
that was observed earlier and **cannot be recovered from its current observation window**. Formally, with W an
obstacle seen before time t − Δ and not visible in o_{t−2..t}:

    memory(Δ) = I( state_t ; W | o_{t−2..t} ) > 0

Three things follow from this definition:
1. **Conditioning on the window is essential.** Information the current image already shows is not memory.
   Every memory measure below therefore needs a control that sees the same current frames: a frame-only probe,
   a novel twin, or the model's own per-frame percept.
2. **Memory has a horizon.** It is a function of the gap Δ, so we report curves against Δ, not a single number.
3. **"Present in the state" is the weakest claim.** We distinguish four levels.

| level | claim | how we test it | can fail even if the lower level holds because |
|---|---|---|---|
| **1. decodable** | the information is in the state | occupancy probe on *memory cells* (seen before, out of view now), against the frame-only CNN and the model's own percept | a probe can read information the predictor never uses (Xie & Alanwar 2026: in the V-JEPA 2 encoder, absent from its predictor) |
| **2. used** | the model's predictions depend on it | memory test E1 / E2: prediction error on pixel-identical re-entry frames, with vs without the earlier sighting | the effect could be carried by something other than the belief (e.g. the state input) |
| **3. causal** | it lives in the belief | memory test E3: swap the belief with the novel twin's, or reset it, at the turn-back | the model may remember, but planning may not exploit it |
| **4. useful** | it improves control | closed-loop CEM on the trap suites with intact / reset belief (deferred, §8) | — |

Levels 1–3 are cheap and diagnostic; level 4 is what ultimately justifies the work.

**The memory cell definition** in the probe is `occ & seen & ~vis`: an occupied ego-centric map cell that has been
inside the occlusion-aware view at some point in the episode but is not visible now. The probe also bins memory
cells by steps since last visible (1–10, …, >160 env steps at 20 Hz), which gives a decodability-vs-age curve.

## 3. Why LeWM has no memory as specified

Details: `docs/lewm_memory_review.md` §2.

| reason | evidence on `kin_v1` |
|---|---|
| **Architecture: no persistent state.** Per-frame embedding plus a 3-frame window (frameskip 5): the predictor sees 0.75 s of the past and nothing else | memory test E1 is exactly 0 at every gap for the baseline and M0 (§6.2) |
| **Objective: memory is not rewarded.** On the original data, only **2.6 %** of the obstacle content of the next frame comes from outside the window but was seen earlier (5.2 % at the 1.25 s CEM horizon). A perfect memory would barely lower the training loss, and 1-step targets give the belief ≈ 0 gradient (Gregor et al. 2019) | `scripts/diag/reobs.py` (§5.1 below) |
| **Representation: the global latent loses geometry, and our state fusion added a shortcut.** Summing the body state into the target embedding made every read-out memorise training layouts | depth decoder 0.09 m train vs 0.62 m held-out; probe visible AP 0.07 vs 0.33 for the frame CNN |

So the absence of memory is a property of the design and the data, not a bug. The test is: **what is the minimal
set of changes that makes a reconstruction-free, SIGReg-regularised latent remember?**

## 4. Hypothesis and model ladder

**Hypothesis.** A reconstruction-free predictive latent keeps out-of-view obstacles if three conditions hold:
- (i) its frame embedding encodes the geometry;
- (ii) it has a persistent state;
- (iii) its objective and data make remembering pay off.

Every reconstruction-free model with measured memory (CPC|Action, PBL, BYOL-Explore, NE-Dreamer) has (ii) and
(iii). Each ladder step adds one change and is compared with the previous step.

| step | change | condition | config / code |
|---|---|---|---|
| baseline | official LeWM + our state fusion `proj(CLS + Embedder(state))` | — | `model=lewm` |
| **M0** | pixels-only embedding `proj(CLS)`; the state enters the predictor as input and is a separate next-state target | (i) remove the shortcut | `model=tok` (`jepa.TokenJEPA`) |
| **M1** | 16 tokens per frame: CLS + the 6×10 patch grid average-pooled to 3×5; SIGReg per token position | (i) keep spatial geometry | `model=tok model.spatial=true` |
| **M2** | recurrent belief replaces the window predictor: closed-loop GRU b_t = GRU(b_{t−1}, [z_t, s_t, a_{t−1}]) (1024-d); open-loop GRU from b_t driven by actions only predicts the frame tokens and state of the next **K = 8 frames (2 s)**; 32-frame (8 s) sequences, batch 16, BPTT through the whole window | (ii) persistent state; (iii) multi-step targets | `model=belief` (`module.Belief`) |
| **M3** | M2 trained on **kin_v2** (look-away glances) with **blind spans**: with p = 0.5 per window, 2–12 consecutive frames have their image replaced by a learned blind vector, so the belief must carry the scene through them | (iii) data and training signal that demand memory | `blind.prob=0.5`, data `kin_v2` |
| **M4** | M3 with a **map belief**: a 32-channel 16×16 ego-centric latent map (0.6 m cells, 9.6 m across), warped every frame by the drone's own motion (body velocity and yaw rate from the state), written through a gated update M_t = (1 − g)·warp(M_{t−1}) + g·tanh(u), read out by a small CNN into the same open-loop head | memory that survives yaw, in a spatial frame | `model.belief._target_=module.MapBelief` |

Fixed across M0–M4:
- ViT-tiny encoder on 48×80 depth (patch 8);
- the BatchNorm MLP projector;
- SIGReg (λ = 0.09, 17 knots, 1024 projections) on frame tokens;
- next-state loss weight 1.0;
- AdamW 5e-5, bf16, 15k steps, seed 3072.

Notes on M2–M4:
- The belief is not regularised by SIGReg. Its effective rank and within-window spread are logged
  (`train/belief_rank`, `belief_time_std`) as collapse monitors.
- The training sequences are 32 frames and the belief starts from zeros at each window. The longest memory that
  is directly trained is therefore ≈ 8 s. The memory test runs the belief over the full rollout (up to 60
  frames), so the longest gaps also test length generalisation.

**SIGReg with tokens.** SIGReg is applied per frame and per token position. M1 has 4 frames × 16 tokens = 64
positions. At batch 128 that exceeds the 5.7 GB GPU: the first backward pass ran out of memory. `token_sigreg`
(`lewm/train.py`) therefore applies it to a random subset of 16 positions per step. That is an unbiased estimate
of the mean over all positions at a quarter of the memory. M0 has only 4 positions, so it was unaffected.

## 5. Data and the memory test

### 5.1 Training data and how much memory it demands

| dataset | content | next frame (+5 steps): content from outside the window but seen before | frames where any such content appears | at +25 steps (1.25 s) |
|---|---|---|---|---|
| `kin_v1` | 1000 episodes, scripted pilot: goal 70 %, explore 15 %, trap entry/exit 15 % | **2.6 %** | 15 % | 5.2 % of content, 20 % of frames |
| `kin_v2` | the same pilot with **look-away glances**: per step, a 1.5 % chance to look 90–150° away from the flight direction for 20–90 steps (1–4.5 s) | **7.9 %** (3×) | 27 % | **22.7 %** of content (4.4×), 42 % of frames |

These are from `scripts/diag/reobs.py` on 150 episodes each. kin_v2 meets the ≥ 10 % target from the review's
risk list at the 1.25 s horizon, though not at one frame. The M2–M4 objective predicts out to 2 s, so it sees
at least the +25 level of demand. Most of the added demand is at
gaps of 0.8–3 s (16–64 steps: 3.5 % vs 0.5 % at +5) and beyond (2.7 % vs 0.6 %).

### 5.2 Label-free memory test (`uav_env/kin_memtest.py`, `lewm/memtest.py`)

**Scenario.** 60 scenarios, `data/memtest_v1`:
1. The drone hovers facing a wall W 1.8–3 m away (3 s).
2. It yaws 110–150° away, so W leaves the 87° view.
3. It holds for a gap Δ ∈ {0, 10, 20, 40, 80, 160} steps (0–8 s).
4. It turns back, W re-enters, and it hovers.

Pillars elsewhere keep the scene non-trivial, and the line of sight to W stays clear. Every scenario is rendered
in four variants:

| variant | W before the gap | W after the gap | identical frames after the gap to |
|---|---|---|---|
| seen | present | present | novel |
| novel | absent | inserted while looking away | seen |
| removed | present | removed while looking away | absent |
| absent | absent | absent | removed |

For each frame j the model predicts the embedding of j from the past only:
- window models: the predictor on the 3 frames before j;
- belief models: the one-step open-loop head from b_{j−1}.

The error is normalised by the target variance. **Re-entry frames** are the first 4 turn-back frames in which W
covers more than 200 depth pixels.

| test | measure | memory predicts |
|---|---|---|
| **E1 memory effect** | e(novel) − e(seen) on re-entry frames, mean ± s.e. over scenarios, against Δ | > 0, decaying with Δ. The frames are pixel-identical, so the only difference is the earlier sighting |
| **E2 surprise / VoE** | e(removed) − e(absent), and voe_acc = P(e(removed) > e(absent)) | > 0 and voe_acc > 0.5: object permanence. The model expected W and it is gone |
| **E3 lesions** (belief models) | seen rollout with the belief at the start of the turn-back replaced by the novel twin's (**swap**) or by zeros (**reset**); belief_share = swap effect / E1 | belief_share ≈ 1: the memory is carried by the belief state |

Notes on the test:
- **It is a within-model contrast.** Absolute re-entry error depends on what the latent encodes, so it is not
  comparable across models. The baseline's 0.10 vs M0's 0.44 reflects the baseline latent being dominated by the
  easy-to-predict body state, not better prediction.
- **Window models give E1 = E2 = 0 by construction.** Nothing before the window reaches them. Measuring exactly 0
  is the check that the twins are truly identical after the gap, so the test cannot leak.
- **Seeds and episodes.** One training seed per model for now. Scenario-level s.e. is reported. Claims need ≥ 3
  seeds (§8).

## 6. Results so far

### 6.1 Occupancy probe (AP; `probe/probe.py`, held-out episodes of `kin_v1`)

Chance rates are 0.009 for visible cells and 0.020 for memory cells. Within one probe run, numbers are directly
comparable. Across runs (rows grouped by run) the spread for the same features is ≈ ±0.001.

| features | what they are | visible | **memory** | never seen |
|---|---|---|---|---|
| *DreamerV3 RSSM (reference)* | recurrent, reconstruction | *0.342* | ***0.105*** | *0.224* |
| frame CNN | current frame only, no memory | 0.329 | 0.062 | 0.175 |
| baseline LeWM `emb` | proj(CLS + state) | 0.070 | 0.058 | 0.170 |
| baseline LeWM CLS | before state fusion | 0.111 | 0.040 | 0.143 |
| baseline LeWM predictor output | 3-frame window | 0.064 | 0.057 | 0.169 |
| baseline LeWM patch tokens | 60 × 192 | 0.278 | 0.063 | 0.184 |
| random-init patch tokens | untrained encoder | 0.310 | 0.065 | 0.180 |
| **M0 percept** | proj(CLS), pixels only | **0.236** | 0.057 | 0.163 |
| **M0 predictor output** | 3-frame window | 0.189 | 0.054 | 0.157 |
| **M1 percept** | 16 tokens (CLS + 3×5 pooled patches), pixels only | 0.208 | 0.054 | 0.161 |
| **M1 predictor output** | 3-frame window, 16 tokens | 0.193 | 0.053 | 0.164 |

Memory AP against steps since last visible (20 Hz; 10 steps = 0.5 s):

| features | 1–10 | 11–20 | 21–40 | 41–80 | 81–160 | > 160 |
|---|---|---|---|---|---|---|
| *DreamerV3 RSSM* | *0.281* | *0.158* | *0.109* | *0.079* | *0.057* | *0.092* |
| frame CNN | 0.151 | 0.053 | 0.041 | 0.034 | 0.032 | 0.068 |
| baseline LeWM `emb` | 0.056 | 0.048 | 0.051 | 0.054 | 0.046 | 0.074 |
| M0 percept | 0.104 | 0.046 | 0.038 | 0.032 | 0.033 | 0.064 |
| M0 predictor output | 0.084 | 0.040 | 0.036 | 0.031 | 0.032 | 0.064 |
| M1 percept | 0.091 | 0.042 | 0.037 | 0.031 | 0.033 | 0.065 |
| M1 predictor output | 0.086 | 0.040 | 0.036 | 0.029 | 0.030 | 0.063 |

What these numbers show:
- **Removing state fusion fixes the representation (M0).** Visible AP rises 0.070 → 0.236. This confirms the
  layout-memorisation diagnosis.
- **No LeWM variant so far carries memory.** Memory AP ≈ 0.057 everywhere, the same as the frame CNN. The
  age-binned curves follow the frame CNN: above chance only at 1–10 steps, where nearby cells are still
  inferable from the current view. The flat ≈ 0.05 of the baseline `emb` across ages is the layout prior that
  the state gives, not memory.
- **The frame CNN beats every frame-level LeWM read-out on visible cells.** The patch tokens (0.278) are close to
  it, but the trained ones are no better than random init (0.310). The geometry is in the pixels; LeWM training
  does not add to it. This is why M1 adds patch tokens rather than relying on the CLS.
- **M1's spatial tokens did not help (prediction falsified, §7).** Visible AP is 0.208 for the percept vs 0.236
  for M0, and 0.193 vs 0.189 for the predictor output. Pooled, projected tokens after JEPA training hold less
  probe-readable geometry than the raw patch tokens of the baseline encoder (0.278). Candidate causes:
  - the shared BatchNorm projector and SIGReg reshaping each token to N(0, I);
  - 2×2 average pooling;
  - a 16× wider probe input (3072-d) overfitting the probe across scenes.

  The last one can be checked by probing the pooled tokens before the projector. M2–M4 build on M1's tokens, so
  this caps the visible-cell geometry available to the belief, but it does not affect the memory test, which
  contrasts a model with itself.
- **Memory AP has a structural floor.** It is never at chance, because part of the memory cells sits next to
  visible structure. The reference for "has memory" is therefore the frame CNN (0.062, and 0.151 at 1–10 steps),
  not 0.02.

### 6.2 Memory test

| model | gaps 0 / 10 / 20 / 40 / 80 / 160 | memory effect | surprise | voe_acc | re-entry error (seen) |
|---|---|---|---|---|---|
| baseline LeWM | all | 0.000 | 0.000 | 0 | 0.100 |
| M0 | all | 0.000 | 0.000 | 0 | 0.437–0.441 |
| M1 | all | 0.000 | 0.000 | 0 | 0.512–0.518 |
| M2–M4 | *(queued)* | | | | |

As expected for window models (§5.2), this is the leak check. The first informative row is M2.

### 6.3 Training curves (validation, at the end of training or latest)

| model | pred loss | state loss | SIGReg | wall time |
|---|---|---|---|---|
| M0 (1 token) | 0.026 | 0.0035 | 2.01 | 2.4 h |
| M1 (16 tokens) | 0.029 | 0.0043 | 2.07 | 3.4 h |

The prediction losses are not comparable across token counts: M1 also predicts 15 spatial tokens.

## 7. What would count as success, per milestone

These are written down before the results land.

| step | prediction | falsified if |
|---|---|---|
| **M1** | visible AP above M0 (0.236), towards the patch-token / frame-CNN level (0.28–0.33); memory AP unchanged; E1 = 0 | visible AP ≤ M0: pooled spatial tokens do not survive the JEPA objective |
| **M2** | **E1 > 0** (≥ 2 s.e.) for gaps up to ~40 steps (2 s, the open-loop horizon), decaying with Δ; belief_share > 0.5; belief memory AP above its own percept and above the frame CNN at 11–40 steps; belief rank stays high | E1 ≈ 0 at all gaps with a healthy belief: multi-step prediction on kin_v1 does not demand memory (§5.1), and M3 becomes the critical test. E1 ≈ 0 with a collapsed rank: an optimisation problem, not a negative result |
| **M3** | E1 extends to longer gaps (80–160 steps); **voe_acc > 0.5**, surprise > 0; memory AP rises in the 21–160 bins | no gain over M2: data and blind spans are not the bottleneck; the belief capacity or the 8 s training window is |
| **M4** | the largest belief_share and the slowest decay with gap; memory AP towards the Dreamer reference (0.105); the decoded map shows W at the right place after the turn | no gain over M3: the GRU already path-integrates, or the warp (from the noisy state) hurts |

An overall positive result is M2 or M3 showing a causal, gap-dependent memory effect (E1 > 0, belief_share ≈ 1)
with memory AP above the frame-only control. The Dreamer reference shows the information is learnable from this
data; matching its 0.105 is not required for the claim.

## 8. Open issues and next steps

- **Seeds.** Every number is a single seed. Rerun the deciding rungs (M2 vs M3, and M4 if it differs) with 3 seeds
  before claiming anything.
- **M2 with K = 1.** The same belief with a one-step head isolates the contribution of multi-step prediction
  (Gregor et al.; BYOL-Explore K = 8 vs 1).
- **Level 4 (closed loop).** The CEM planner first needs a fix: a value head on b_t or a longer horizon. Its
  current failures are planner-limited (1.25 s horizon, no value). Then run the trap suites with intact vs reset
  belief, within the LeWM variants only.
- **Training window vs test gaps.** The belief is trained on 8 s windows from a zero start. If E1 vanishes at the
  80–160-step gaps, the next experiment is longer windows or carrying the belief across windows (truncated BPTT
  with state carry).
- **Belief collapse.** Watch `belief_rank` and `belief_time_std`. If they collapse, apply SIGReg to the belief,
  which is the untested "first SIGReg-regularised recurrent JEPA belief" in the literature gap.
- **Stand-in simulator.** Everything is on analytic depth. Isaac Sim confirmation waits on the driver fix.

## 9. Relevance to the multi-agent world-model project

The single-agent question is the prerequisite for the multi-agent one:
- **What agents communicate is memory.** One agent's knowledge of places another cannot see. If a single agent's
  latent holds nothing beyond its current view, there is nothing worth sending.
- **E3 is the multi-agent experiment in miniature.** Swapping in another rollout's belief and watching
  predictions follow is the same operation as injecting a teammate's belief.
- **M4's map belief is a natural message format.** It lives in an explicit spatial frame, so another agent's map
  can be warped by the relative pose and fused.
- **The component analysis says what to transmit.** It locates where memory lives: percept tokens, the state
  path, the belief or the map.
- **The decay curve against Δ says how stale a message may be.**

Planned follow-ups: rate-limited beliefs (bits vs E1), frame equivariance and belief fusion. See
`docs/lewm_memory_review.md` §5.3.

## 10. Reproducing

```bash
# data (CPU)
SIM=kin bash scripts/lewm_memory.sh memtest_data collect_v2
# models: each trains 15k steps, then extracts probe features, runs the memory test and the probe
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/run_frozen.sh lewm_chain m0 m1 m2 m3 m4
# memory demand of a dataset (CPU)
source env.sh; PYTHONPATH=uav_env $PY_MAIN scripts/diag/reobs.py data/kin_v2 150
```

Per model the outputs are:
- `runs/kin_v1/<m>/`: `lewm_object.ckpt`, `train_metrics.jsonl`, `memtest.json`, `memtest_traces.npz`, and
  `features_percept` plus `features_pred` or `features_belief`;
- `runs/kin_v1/probe_<m>/`: `probe_results.json` and figures.

Figures after M4:
- `scripts/viz/memory_figures.py`, `latent_figures.py`;
- `render_memtest.py`;
- `belief_figures.py traj | map`;
- `render_episode.py` with `VIZ_SETS`.
