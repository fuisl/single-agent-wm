# Single-agent memory in a latent-predictive world model (R-LeWM)

Status: 2026-10-22 11:30. M0–M4 done. **M2–M4 have a collapsed frame encoder** (§6.4), so their memory results
are not yet evidence either way. The fix (§6.5, pre-registered) is training as M2s, then M3s / M4s if M2s passes
the gate (tmux `lewm_gpu`, log `logs/lewm_sigfix.log`). Weights: huggingface.co/fuisloy/r-lewm-uav.
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
| **M2 percept** | 16 tokens, trained with the belief | 0.112 | 0.041 | 0.138 |
| **M2 belief** | closed-loop GRU state b_t (1024-d) | 0.065 | **0.059** | 0.168 |
| M3 percept | M2 on kin_v2 with blind spans (collapsed encoder, §6.4) | 0.053 | 0.033 | 0.123 |
| M3 belief | | 0.038 | 0.055 | 0.157 |
| M4 percept | M3 with the map belief (collapsed encoder, §6.4) | 0.055 | 0.033 | 0.123 |
| M4 belief | ego-centric latent map read-out | 0.038 | 0.055 | 0.156 |

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
| M2 percept | 0.059 | 0.029 | 0.027 | 0.026 | 0.028 | 0.057 |
| **M2 belief** | 0.059 | **0.054** | **0.052** | **0.054** | 0.047 | 0.076 |

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
- **M2's belief holds more about out-of-view cells than its own percept, but not much more than a single frame.**
  - Memory AP is 0.059 for the belief against 0.041 for the percept.
  - By age, the belief stays flat at ≈ 0.053 from 11 to 80 steps, where the percept falls to ≈ 0.027. That is
    level with the frame CNN at 11–20 steps (0.053) and above it at 21–80 steps (0.041 and 0.034).
  - Two caveats:
    - The belief also takes the state as input. Its age curve lies almost exactly on the baseline `emb` curve
      (0.048–0.054 from 11 to 160 steps), which §6.1 attributes to the layout prior that the state gives. So the
      probe gain is most likely the state, not memory, until a state-ablated probe shows otherwise. The memory
      test (§6.2) has no such confound, because the twins share the state.
    - M2's percept lost geometry: visible AP fell from 0.208 (M1) to 0.112. Two things changed at once: the
      gradient now comes only through the belief, and the batch is 16 windows of 32 frames instead of 128 windows
      of 4.
- **Memory AP has a structural floor.** It is never at chance, because part of the memory cells sits next to
  visible structure. The reference for "has memory" is therefore the frame CNN (0.062, and 0.151 at 1–10 steps),
  not 0.02.

### 6.2 Memory test

| model | gaps 0 / 10 / 20 / 40 / 80 / 160 | memory effect | surprise | voe_acc | re-entry error (seen) |
|---|---|---|---|---|---|
| baseline LeWM | all | 0.000 | 0.000 | 0 | 0.100 |
| M0 | all | 0.000 | 0.000 | 0 | 0.437–0.441 |
| M1 | all | 0.000 | 0.000 | 0 | 0.512–0.518 |
| M2 | see below | 0.0005 – 0.0021 | 0.0013 – 0.0047 | 0.62 – 0.93 | 0.531–0.546 |
| M3 | all | **−0.033 → −0.007** (−3.5 to −3.7 s.e.) | 0.050 → 0.008 (3.8–4.6 s.e.) | 0.67–0.82 | 0.827–0.841 |
| M4 | all | +0.003 → −0.005 (all < 2 s.e.) | 0.026 → 0.001 (3.0–4.6 s.e.) | 0.62–0.80 | **0.667 → 2.017** |

As expected for window models (§5.2), the first three rows are the leak check. M2 is the first informative row.
The M3 and M4 rows come from collapsed encoders (§6.4), and their normaliser (target variance on the memtest
frames) is 0.0035–0.0042 against 0.090 for M2, so their normalised numbers are not comparable with M2's.
Its errors are normalised by the target variance; the bracket is ± 1 s.e. over the 60 scenarios.

| M2, gap (steps) | E1 memory effect | E2 surprise | voe_acc | reset effect |
|---|---|---|---|---|
| 0 | 0.0005 ± 0.0009 (0.5 s.e.) | 0.0044 ± 0.0010 (4.2 s.e.) | 0.62 | 0.030 |
| 10 | 0.0013 ± 0.0008 (1.7 s.e.) | 0.0047 ± 0.0009 (5.0 s.e.) | 0.77 | 0.030 |
| 20 | **0.0020 ± 0.0007 (2.9 s.e.)** | 0.0040 ± 0.0006 (6.3 s.e.) | 0.80 | 0.032 |
| 40 | **0.0021 ± 0.0005 (3.8 s.e.)** | 0.0033 ± 0.0005 (6.7 s.e.) | 0.85 | 0.031 |
| 80 | **0.0017 ± 0.0003 (4.8 s.e.)** | 0.0023 ± 0.0003 (7.3 s.e.) | 0.93 | 0.031 |
| 160 | **0.0010 ± 0.0002 (6.1 s.e.)** | 0.0013 ± 0.0002 (8.1 s.e.) | 0.90 | 0.029 |

What M2 shows:
- **The recurrent belief carries a real memory effect.**
  - E1 is above 2 s.e. from gap 20 to gap 160 (1 s to 8 s out of view, plus 1.5 s of turning).
  - The re-entry frames are pixel-identical and the states are the same, so the difference can only come from
    the earlier sighting.
  - The effect is **small**: 0.002 is ≈ 0.4 % of the re-entry error (0.54).
- **Object permanence (E2) is the clearer signal.**
  - Removing a wall the model saw raises its error at every gap (4–8 s.e.).
  - The removed twin has the higher error in 62–93 % of scenarios (voe_acc).
- **The profile is not the one predicted.**
  - E1 is not significant at gaps 0–10 and peaks at 20–40.
  - Both E1 and E2 shrink from 40 to 160 steps, while voe_acc rises with gap.
  - Longer gaps put the scenarios at the edge of the 8 s training window or beyond it (§8). At those gaps the
    errors are smaller and less variable, which may explain the high voe_acc.
- **E3 is not informative for M2.**
  - belief_share = 1.000 at every gap. This holds by construction: the GRU state is the only path from the past.
    Swapping it at the turn-back therefore reproduces the novel run exactly.
  - Reset (belief → 0) costs 0.03, 15× the memory effect, so the belief carries much more than the wall: pose,
    scene layout and motion context.
  - The lesion becomes a real test in M4, where it acts on the map alone.

### 6.3 Training curves (validation, at the end of training or latest)

| model | pred loss | state loss | SIGReg | wall time |
|---|---|---|---|---|
| M0 (1 token) | 0.026 | 0.0035 | 2.01 | 2.4 h |
| M1 (16 tokens) | 0.029 | 0.0043 | 2.07 | 3.4 h |
| M2 (belief) | 0.017 (8-step mean), 0.0059 (1-step) | 0.0072 | 1.99 | 3.0 h (+ 0.6 h eval) |
| M3 (kin_v2, blind spans) | 0.020 / 0.012 | 0.0082 | 2.10 | 3.3 h |
| M4 (map belief) | 0.021 / 0.013 | 0.0083 | 2.08 | 3.5 h |

The prediction losses are not comparable across token counts: M1 also predicts 15 spatial tokens. The SIGReg
values are not comparable across batch sizes either (§6.4): M0/M1 estimate it from 128 samples, M2–M4 from 16.

### 6.4 The frame encoder of M2–M4 collapsed

Found on 2026-10-22, after M3 and M4 finished. The belief models' collapse monitors watched only the belief
(rank 71–101, healthy), not the frame embedding the belief has to remember. Measured on the CLS token
(`lewm/percept_diag.py`; 1600 frames of 60 kin_v1 validation episodes, 60 memtest scenarios):

| | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| effective rank (RankMe) on kin_v1 frames | 74.7 | 68.2 | **8.6** | **6.4** | **6.6** |
| variance share of the top direction | 0.04 | 0.04 | 0.30 | 0.32 | 0.31 |
| wall signal: ‖z(W in view) − z(no W)‖², same frames otherwise | 1.49 | 1.44 | 0.63 | **0.010** | **0.011** |
| SIGReg statistic at N = 16 samples (M2–M4 training) | 1.08 | 1.11 | 1.97 | 2.09 | 2.05 |
| SIGReg statistic at N = 128 samples (M0/M1 training) | 1.92 | 2.00 | **9.9** | **11.4** | **11.2** |

An i.i.d. N(0, I) sample scores 1.02–1.05 at every N.

What this means:
- **The encoder of every belief model lost most of its dimensions**: rank 66–73 → 6–9.
- **In M3 and M4 the embedding barely registers the wall** (wall signal ~150× smaller than M0/M1). There is little
  for the belief to remember, so E1 ≈ 0 or of the wrong sign, and low probe AP, follow regardless of the belief.
- M2's geometry loss (visible AP 0.208 → 0.112, §6.1) has the same cause: the M2 encoder is already at rank 8.6.

**Why SIGReg did not prevent it.** The Epps–Pulley statistic in SIGReg is
EP = N ∫ |φ̂_N(t) − φ(t)|² w(t) dt, with φ̂_N the empirical characteristic function of N samples (LeJEPA §4.2.3;
the factor N is in the official code, `module.SIGReg`). Under the null its expectation is a constant. A deviation Δ
from N(0, I) costs ≈ N·Δ. The penalty on collapse therefore scales with N, at fixed λ.
- LeWM and our M0/M1 estimate it per frame over the batch, N = 128.
- `token_sigreg` in M2–M4 also takes N = B = 16, because the batch was cut from 128 windows of 4 frames to 16 of
  32 to fit BPTT in memory. At the same λ = 0.09 the anti-collapse pressure is ≈ 8× weaker.
- At N = 16 the collapsed encoders score 2.0–2.1 against M0's 1.08, so the logged SIGReg (≈ 2) looked normal. At
  N = 128 they score 10–11 against M0's 1.92.

Why the belief models collapse while M0/M1 do not, at the same encoder: the 8-step open-loop target is harder to
predict than the next frame. JEPA objectives favour predictable, slow features (Littwin et al. 2024; Sobal et al.
2022; H-JEPA App. L), and an 8× weaker regulariser lets the encoder drop what it cannot predict.

Two other M3/M4 observations, to revisit only after the fix, because both rest on the collapsed encoder:
- **M3's E1 is negative** at every gap: having seen W makes re-entry prediction worse, and zeroing the belief at
  the turn-back lowers the error (reset effect −0.06 to −0.03).
- **M4's error grows with the gap** (0.67 at 0 → 2.02 at 160 steps), and zeroing the map at the turn-back lowers it
  by up to 1.31. The warped map accumulates error beyond the 8 s training window.

### 6.5 The fix, its basis, and what it predicts (written before the M2s results)

**What the literature says.**

| source | relevant finding | consequence for us |
|---|---|---|
| LeJEPA (Balestriero & LeCun 2025, arXiv:2511.08544) §4.2.3, §5.1, Thm 6, Tab. 1c | EP statistic carries the factor N. Minibatch bias of loss and gradient is O(1/N), "not a concern even for minibatches as small as 16". Results are reported for batch 128–1024 (ViT-L: 72.2–74.7 %), and the recommended starting point is ≥ 128 | the bias argument does not cover effective strength; our N = 16 is outside every tested setting |
| LeWM (Maes et al. 2026, arXiv:2603.19312) App. D, G, H | batch 128 × 4 frames; SIGReg per time step over the batch; λ ∈ [0.01, 0.2] all > 80 % success at batch 128, peak near 0.09; per-step SIGReg leaves the temporal structure unconstrained | our λ = 0.09 is tuned for N = 128 |
| MotionJEPA (Karmann et al. 2026, arXiv:2609.23881) Tab. 1a | *LeWM-Flat*, SIGReg over the flattened batch × time samples, probes better than per-step LeWM in all three games (NMSE 0.284 / 0.331 / 0.694 vs 0.658 / 0.394 / 0.763). SIGReg along time only (*LeWM-Time*) is worse and "completely fails" on Golf | pooling frames into the sample is a tested, better-than-default placement; time-only is not |
| RankMe (Garrido et al., ICML 2023, arXiv:2210.02885) | the effective rank of the embeddings, exp of the entropy of normalised singular values, is a label-free predictor of downstream quality in joint-embedding SSL | the percept monitor we lacked; the same formula as our `belief_rank` |
| Jing et al., ICLR 2022 (arXiv:2110.09348) | joint-embedding methods can collapse dimensionally (low-rank spectrum) while not collapsing completely | what we see: SIGReg ≈ 2, no constant output, rank 7 |
| Liu et al. 2026, TC-SIGReg (arXiv:2607.26924) | raw LeWM SIGReg "biases variance allocation toward the temporally persistent component"; SIGReg on temporally centred residuals fixes it for policy learning | a different failure (allocation, not rank). Not adopted now: a candidate if the fixed M2s keeps the rank but still drops the wall |

**The change** (`loss.sigreg.samples=128`, `token_sigreg` in `lewm/train.py`). Per token position, pool the B × T
frames of the batch and draw 128 of them for the statistic.
- **Same estimator and strength as the runs that did not collapse.** N = 128 as in LeWM and M0/M1, with λ = 0.09
  unchanged. Memory cost is unchanged from M1 (16 positions × 128 samples).
- **The same target distribution.** Windows start at random offsets, so the marginal of a frame embedding does not
  depend on its index in the window. Pooling frames estimates the same N(0, I) marginal from more samples. Frames
  of one window are correlated, which raises the O(1/N) bias towards O(1/N_eff) with N_eff ≥ 16, the level LeJEPA
  calls negligible.
- **Rejected alternatives.**
  - λ × 8 at N = 16: the same expected penalty, but from a 16-sample estimate, so a noisier gradient and a larger
    bias.
  - A larger batch: does not fit BPTT over 32 frames in 5.7 GB.
  - SIGReg along time: worse in MotionJEPA.
  - TC-SIGReg: targets a different failure (above).
- **Monitor.** `train/percept_rank` (RankMe of the CLS over the batch frames) is now logged for every model.
  `lewm/percept_diag.py` (rank, SIGReg at N = 16/128, wall signal) runs in `evaluate` after every model.

**Pre-registered predictions for M2s** (M2 + the fix, otherwise identical: kin_v1, seed 3072, 15k steps):

| quantity | M2 | prediction for M2s | if not |
|---|---|---|---|
| CLS rank on kin_v1 (percept_diag) | 8.6 | **≥ 50** (M0 75, M1 68) | the collapse has another cause (objective pressure); stop and rethink, do not run M3s/M4s |
| SIGReg at N = 128 (percept_diag) | 9.9 | **≤ 3** (M0 1.92) | the fix did not reach the encoder |
| visible AP, percept | 0.112 | ≥ 0.18 (M1 0.208) | rank is back but geometry is not: the 8-step target, not the regulariser, discards it (then TC-SIGReg / a 1-step percept loss) |
| wall signal | 0.63 | ≥ 1.0 (M1 1.44) | as above |
| E1 relative to re-entry error | 0.4 % | larger than M2 at gaps 20–80 | the belief, not the encoder, is the bottleneck for memory |

**Gate.** M3s and M4s (M3 and M4 with the fix) run only if M2s meets the first two rows. The gate is the stage
`gate_m2s` in `scripts/lewm_memory.sh`, and it fails the chain loudly otherwise. M3s and M4s are then judged against
the M3 and M4 predictions of §7.

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
