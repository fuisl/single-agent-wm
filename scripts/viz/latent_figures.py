"""Static figures for the LeWM memory work (one message per figure), into runs/kin_v1/viz/figures/.

    source env.sh
    $PY_MAIN scripts/viz/latent_figures.py

  latent_pca.png      2-D PCA of LeWM's final emb / ViT CLS (before the state is added) / pooled patch tokens over
                      val episodes, coloured by episode, with the share of variance explained by the episode
                      identity and a same-episode nearest-neighbour rate (high = the latent identifies the layout)
  decoders.png        depth decoded from each LeWM read-out, train vs held-out layouts (runs/.../lewm/diag.json)
  memory_demand.png   where the obstacle content of a future frame was last available (scripts/diag/reobs.py)
  probe_groups.png    probe AP for visible / remembered / never-seen cells, every feature set in runs/.../probe*/
  probe_age.png       probe AP on remembered cells vs time since they were last visible
Inputs that do not exist yet are skipped with a note.
"""
import argparse
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p = argparse.ArgumentParser()
p.add_argument("--run", default=os.path.join(ROOT, "runs/kin_v1"))
p.add_argument("--n_episodes", type=int, default=20)
p.add_argument("--out", default=None)
args = p.parse_args()
OUT = args.out or os.path.join(args.run, "viz/figures")
os.makedirs(OUT, exist_ok=True)
# Okabe-Ito (colour-blind safe); grey is kept for the DreamerV3 reference
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
REF = "#7f7f7f"
LABELS = {"wm": "DreamerV3 (reference)", "frame": "current frame only (CNN)", "lewm": "LeWM 3-frame ctx",
          "lewm_emb": "LeWM final emb", "lewm_rand": "LeWM untrained", "rand": "DreamerV3 untrained",
          "lewm_cls": "LeWM ViT CLS", "lewm_patch": "LeWM patch tokens", "lewm_rand_patch": "LeWM untrained patch",
          "lewm_pred": "LeWM predictor output", "lewm_pred_h": "LeWM predictor hidden"}
def label(k):
    """Display name of a probe feature set; m<i>_<part> names come from scripts/lewm_memory.sh evaluate()."""
    if k in LABELS:
        return LABELS[k]
    m, _, part = k.partition("_")
    return f"{m.upper()} {part}" if m[:1] == "m" and m[1:].isdigit() else k


plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 13})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name), flush=True)


# ---------------------------------------------------------------- latent PCA by episode
def latent_pca():
    reps = [("emb", "final emb = proj(CLS + state)"), ("cls", "ViT CLS (pixels only)"),
            ("patch", "patch tokens (pooled 3x5)")]
    dirs = {r: os.path.join(args.run, "lewm", f"features_{r}") for r, _ in reps}
    reps = [(r, t) for r, t in reps if os.path.isdir(dirs[r])]
    if not reps:
        print("note: no LeWM feature dirs, skipping latent_pca.png"); return
    names = None
    for r, _ in reps:
        have = {os.path.basename(f) for f in glob.glob(os.path.join(dirs[r], "*.npz"))
                if str(np.load(f)["split"]) == "val"}
        names = have if names is None else names & have
    names = sorted(names)[:args.n_episodes]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, len(reps), figsize=(5.2 * len(reps), 5))
    ax = np.atleast_1d(ax)
    for a, (r, title) in zip(ax, reps):
        X, ep, tt = [], [], []
        for i, n in enumerate(names):
            f = np.load(os.path.join(dirs[r], n))["feat"].astype(np.float32)
            X.append(f); ep.append(np.full(len(f), i)); tt.append(np.arange(len(f)))
        X, ep, tt = np.concatenate(X), np.concatenate(ep), np.concatenate(tt)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        # share of variance explained by the episode identity (correlation ratio, averaged over dimensions)
        means = np.stack([X[ep == i].mean(0) for i in range(len(names))])
        eta2 = float((np.bincount(ep)[:, None] * means ** 2).sum() / (X ** 2).sum())
        # same-episode nearest neighbour, excluding the frame's own +-20-step neighbourhood
        sub = rng.choice(len(X), min(2500, len(X)), replace=False)
        Xs = X[sub]
        d2 = (Xs ** 2).sum(1)[:, None] + (Xs ** 2).sum(1)[None] - 2 * Xs @ Xs.T
        same = ep[sub][:, None] == ep[sub][None]
        d2[same & (np.abs(tt[sub][:, None] - tt[sub][None]) <= 20)] = np.inf
        nn_same = float(same[np.arange(len(sub)), d2.argmin(1)].mean())
        U, S, Vt = np.linalg.svd(X[sub] - X[sub].mean(0), full_matrices=False)
        P = (X - X[sub].mean(0)) @ Vt[:2].T
        cmap = plt.get_cmap("tab20")
        a.scatter(P[:, 0], P[:, 1], c=[cmap(i % 20) for i in ep], s=2, alpha=0.5, linewidths=0)
        a.set_title(f"{title}\nvariance explained by episode {eta2:.0%}, same-episode NN {nn_same:.0%}", fontsize=11)
        a.set_xticks([]); a.set_yticks([]); a.set_xlabel("PC 1"); a.set_ylabel("PC 2")
    fig.suptitle(f"LeWM latents of {len(names)} held-out episodes, one colour per episode "
                 f"(chance same-episode NN ≈ {1 / len(names):.0%})", fontsize=13)
    fig.tight_layout()
    save(fig, "latent_pca.png")


# ---------------------------------------------------------------- decoders (diag.json)
def decoders():
    f = os.path.join(args.run, "lewm", "diag.json")
    if not os.path.exists(f):
        print(f"note: no {f}, skipping decoders.png"); return
    dec = json.load(open(f))["decode"]
    order = [("trained/patch", "patch tokens\n(60 x 192)"), ("random/patch", "patch tokens\nuntrained"),
             ("trained/cls", "ViT CLS\n(pixels only)"), ("random/emb", "final emb\nuntrained"),
             ("trained/emb", "final emb\n(CLS + state)"), ("trained/state", "state\nembedding only")]
    order = [(k, l) for k, l in order if k in dec]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    tr = [dec[k]["train"]["mae_m"] for k, _ in order]
    va = [dec[k]["val"]["mae_m"] for k, _ in order]
    ax.bar(x - 0.2, tr, 0.4, color=OI[5], label="train episodes")
    ax.bar(x + 0.2, va, 0.4, color=OI[0], label="held-out layouts")
    for i, v in enumerate(va):
        ax.text(x[i] + 0.2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(1.03, color=REF, ls="--", lw=1)
    ax.text(len(order) - 0.5, 1.04, "constant mean image", color=REF, ha="right", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in order], fontsize=10)
    ax.set_ylabel("depth MAE (m)")
    ax.set_title("The encoder keeps the geometry; the global embedding loses it,\nand adding the state makes it memorise training layouts")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "decoders.png")


# ---------------------------------------------------------------- memory demand (reobs)
def memory_demand():
    # scripts/diag/reobs.py on data/kin_v1 (60 episodes; trap = 60 trap episodes), docs/lewm_memory_review.md §2.2
    rows = {"next frame (+0.25 s)": [0.909, 0.067, 0.014, 0.005, 0.005],
            "next frame, trap episodes": [0.921, 0.056, 0.013, 0.004, 0.005],
            "planning horizon (+1.25 s)": [0.668, 0.287, 0.016, 0.011, 0.017],
            "planning horizon, trap episodes": [0.717, 0.241, 0.015, 0.009, 0.017]}
    cats = ["in LeWM's 3 context frames", "never seen before", "seen 0.05-0.75 s ago",
            "seen 0.8-3.2 s ago", "seen > 3.2 s ago"]
    cols = [OI[5], "#d9d9d9", OI[1], OI[3], OI[4]]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for j, (c, col) in enumerate(zip(cats, cols)):
        v = np.array([r[j] for r in rows.values()])
        ax.barh(y, v, left=left, color=col, label=c)
        left += v
    mem = [sum(r[3:]) for r in rows.values()]
    for yi, m in zip(y, mem):
        ax.text(1.01, yi, f"older than the\n0.75 s window: {m:.1%}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(list(rows)); ax.set_xlim(0, 1)
    ax.set_xlabel("share of obstacle cells visible in the target frame")
    ax.set_title("A 1-step objective almost never asks for memory: obstacles in the next frame are already in context")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=10)
    save(fig, "memory_demand.png")


# ---------------------------------------------------------------- probe summaries
def probe_results():
    res = {}
    for f in sorted(glob.glob(os.path.join(args.run, "probe*", "probe_results.json"))):
        for k, v in json.load(open(f)).items():
            res.setdefault(k, v)                     # first run wins (dedupes the frame baseline)
    if not res:
        print("note: no probe_results.json, skipping probe figures")
    return res


def colour(k, i):
    return {"wm": REF, "rand": "#c7c7c7"}.get(k, OI[i % len(OI)])


def probe_groups(res):
    keys = [k for k in LABELS if k in res] + [k for k in res if k not in LABELS]
    groups = [("vis", "visible now"), ("mem", "seen, now out of view"), ("unseen", "never seen")]
    w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, k in enumerate(keys):
        v = [res[k][g]["ap"] for g, _ in groups]
        ax.bar(np.arange(3) + (i - (len(keys) - 1) / 2) * w, v, w, color=colour(k, i), label=label(k),
               hatch="//" if k in ("wm", "rand") else None, edgecolor="white")
    base = [res[keys[0]][g]["base_rate"] for g, _ in groups]
    for j, b in enumerate(base):
        ax.plot([j - 0.42, j + 0.42], [b, b], color="black", lw=1, ls=":")
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in groups])
    ax.set_ylabel("obstacle average precision (dotted = chance)")
    ax.set_title("What a probe can read out of each latent (held-out episodes)")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    save(fig, "probe_groups.png")


def probe_age(res):
    keys = [k for k in LABELS if k in res] + [k for k in res if k not in LABELS]
    bins = list(res[keys[0]]["mem_by_age"])
    lab = [f"{int(b.split('-')[0]) / 20:g}-{int(b.split('-')[1]) / 20:g} s" for b in bins]
    lab[-1] = f"> {int(bins[-1].split('-')[0]) / 20:g} s"
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, k in enumerate(keys):
        ax.plot(range(len(bins)), [res[k]["mem_by_age"][b]["ap"] for b in bins], marker="o", color=colour(k, i),
                ls={"wm": "--", "rand": ":"}.get(k, "-"), lw=2.2 if k == "wm" else 1.6, label=label(k))
    ax.plot(range(len(bins)), [res[keys[0]]["mem_by_age"][b]["base_rate"] for b in bins], color="black", ls=":",
            lw=1, label="chance")
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(lab)
    ax.set_xlabel("time since the obstacle cell was last visible")
    ax.set_ylabel("obstacle AP on remembered cells")
    ax.set_title("Memory of out-of-view obstacles fades within seconds")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    save(fig, "probe_age.png")


latent_pca()
decoders()
memory_demand()
res = probe_results()
if res:
    probe_groups(res)
    probe_age(res)
