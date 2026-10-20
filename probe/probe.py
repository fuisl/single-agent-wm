"""Memory probe: can a small decoder read the ego-centric occupancy map out of the RSSM state h_t?

Runs in the main env (.venv, torch + scenegen):
    python probe.py --data data/wm_v1 --feats wm=runs/wm50m/features rand=runs/wm50m/features_rand \
                    --out runs/wm50m/probe

Feature sets are probed with the same MLP; a "frame" baseline probes from the current depth image only
(a CNN, no memory). Evaluation on held-out episodes splits cells into
  vis    = visible now (perception),
  mem    = seen earlier but out of view now (memory),
  unseen = never seen (guessing),
and reports IoU / recall of obstacle cells per group, plus memory recall vs. steps since last seen.
Also writes the open-loop imagination error vs. horizon from the extracted features.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn

import scenegen as sg

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--feats", nargs="+", required=True, help="name=dir pairs")
p.add_argument("--out", required=True)
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--stride", type=int, default=2, help="use every n-th training step")
p.add_argument("--no_frame", action="store_true")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda"
AGE_BINS = [(1, 10), (11, 20), (21, 40), (41, 80), (81, 160), (161, 10_000)]


def is_val(name, frac=0.1):
    """Default train/val split used everywhere: 10% of episodes by a hash of the file name."""
    import zlib
    return (zlib.crc32(os.path.basename(name).encode()) % 1000) < frac * 1000


# ---------------------------------------------------------------- labels (+ age since last visible)
def ages_for(d):
    """Steps since each ego cell was last visible (0 = visible now, large = never)."""
    boxes = d["boxes"].astype(np.float64)
    last = np.full((sg.SEEN_N, sg.SEEN_N), -10_000, np.int32)
    X, Y = np.meshgrid(*(2 * [-sg.EXTENT + (np.arange(sg.SEEN_N) + 0.5) * sg.SEEN_RES]), indexing="ij")
    out = np.zeros((len(d["pos"]), sg.EGO_N, sg.EGO_N), np.int32)
    w = int(np.ceil((sg.D_MAX + 0.3) / sg.SEEN_RES))
    for t, (pos, yaw) in enumerate(zip(d["pos"].astype(np.float64), d["yaw"].astype(np.float64))):
        dirs = np.stack([np.cos(yaw + sg.RAY_ANG), np.sin(yaw + sg.RAY_ANG)], -1)
        rr = sg.raycast2d(pos[:2], dirs, boxes, sg.D_MAX + 1.0)
        ci, cj = ((pos[:2] + sg.EXTENT) / sg.SEEN_RES).astype(int)
        i0, i1, j0, j1 = max(ci - w, 0), min(ci + w + 1, sg.SEEN_N), max(cj - w, 0), min(cj + w + 1, sg.SEEN_N)
        if i0 < i1 and j0 < j1:
            pts = np.stack([X[i0:i1, j0:j1], Y[i0:i1, j0:j1]], -1)
            sub = last[i0:i1, j0:j1]
            sub[sg.visibility(pts, pos, yaw, rr)] = t
        ij = np.clip(np.floor((sg.ego_points(pos, yaw) + sg.EXTENT) / sg.SEEN_RES).astype(int), 0, sg.SEEN_N - 1)
        out[t] = t - last[ij[..., 0], ij[..., 1]]
    return np.minimum(out, 10_000)


def load(name_dirs):
    first = list(name_dirs.values())[0]
    names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(first, "ep_*.npz")))
    eps = []
    cache_dir = os.path.join(args.out, "label_cache"); os.makedirs(cache_dir, exist_ok=True)
    for i, n in enumerate(names):
        d = np.load(os.path.join(args.data, n))
        cpath = os.path.join(cache_dir, n)
        if os.path.exists(cpath):
            age = np.load(cpath)["age"]
        else:
            age = ages_for(d); np.savez_compressed(cpath, age=age)
        f0 = np.load(os.path.join(first, n))
        split = str(f0["split"]) if "split" in f0.files else ("val" if is_val(n) else "train")
        ep = {"name": n, "split": split, "occ": d["occ"], "seen": d["seen"], "vis": d["vis"],
              "age": age, "depth": d["depth"]}
        for k, dd in name_dirs.items():
            f = np.load(os.path.join(dd, n))
            # any world model: either a single "feat" array (T, D) or DreamerV3's deter + stoch
            ep[k] = f["feat"] if "feat" in f.files else np.concatenate([f["deter"], f["stoch"]], -1)
        eps.append(ep)
        if (i + 1) % 100 == 0:
            print(f"labels {i + 1}/{len(names)}", flush=True)
    return eps


# ---------------------------------------------------------------- probes
class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, 1024), nn.SiLU(), nn.Dropout(0.2),
                                 nn.Linear(1024, 1024), nn.SiLU(),
                                 nn.Linear(1024, sg.EGO_N * sg.EGO_N))

    def forward(self, x):
        return self.net(x).view(-1, sg.EGO_N, sg.EGO_N)


class FrameProbe(nn.Module):
    def __init__(self):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.SiLU())
        self.cnn = nn.Sequential(c(1, 32), c(32, 64), c(64, 128), c(128, 128), nn.Flatten())   # 3 x 5
        self.head = nn.Sequential(nn.Linear(128 * 15, 1024), nn.SiLU(), nn.Linear(1024, sg.EGO_N * sg.EGO_N))

    def forward(self, x):
        x = (x.float()[:, None] - sg.D_MIN) / (sg.D_MAX - sg.D_MIN)
        return self.head(self.cnn(x)).view(-1, sg.EGO_N, sg.EGO_N)


def stack(eps, key, stride=1):
    return np.concatenate([e[key][::stride] for e in eps], 0)


def fit(model, X, Y, epochs, Xv, Yv):
    """AdamW + dropout; keeps the epoch with the lowest loss on held-out training episodes (early stopping)."""
    import copy
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.05)
    pos_w = torch.tensor(10.0, device=dev)
    lossf = lambda x, y: nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_w)
    n, best, best_state, bad = len(X), np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, 512):
            idx = perm[k:k + 512].numpy()
            x = torch.as_tensor(X[idx], device=dev).float()
            y = torch.as_tensor(Y[idx], device=dev).float()
            loss = lossf(x, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = np.mean([lossf(torch.as_tensor(Xv[k:k + 1024], device=dev).float(),
                                torch.as_tensor(Yv[k:k + 1024], device=dev).float()).item()
                          for k in range(0, len(Xv), 1024)])
        print(f"  epoch {ep + 1}: train {tot / n:.4f}  held-out {vl:.4f}", flush=True)
        if vl < best:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = []
    for k in range(0, len(X), 1024):
        out.append(torch.sigmoid(model(torch.as_tensor(X[k:k + 1024], device=dev).float())).cpu().numpy())
    return np.concatenate(out)


def avg_precision(score, label, bins=1000):
    """Threshold-free average precision via a score histogram (exact up to bin width)."""
    h_pos = np.bincount(np.minimum((score[label] * bins).astype(int), bins - 1), minlength=bins)[::-1]
    h_neg = np.bincount(np.minimum((score[~label] * bins).astype(int), bins - 1), minlength=bins)[::-1]
    tp, fp = np.cumsum(h_pos), np.cumsum(h_neg)
    if tp[-1] == 0:
        return 0.0
    prec = tp / np.maximum(tp + fp, 1)
    return float((prec * h_pos).sum() / tp[-1])


def scores(pred, occ, seen, vis, age):
    P = pred > 0.5
    groups = {"vis": vis, "mem": seen & ~vis, "unseen": ~seen}
    res = {}
    for g, m in groups.items():
        tp = (P & occ & m).sum(); fp = (P & ~occ & m).sum(); fn = (~P & occ & m).sum()
        res[g] = {"ap": avg_precision(pred[m], occ[m]), "base_rate": float(occ[m].mean()),
                  "iou": float(tp / max(tp + fp + fn, 1)), "recall": float(tp / max(tp + fn, 1)),
                  "precision": float(tp / max(tp + fp, 1)), "obstacle_cells": int((occ & m).sum())}
    mem = seen & ~vis
    res["mem_by_age"] = {}
    for lo, hi in AGE_BINS:
        m = mem & (age >= lo) & (age <= hi)
        tp = (P & occ & m).sum(); fp = (P & ~occ & m).sum(); fn = (~P & occ & m).sum()
        res["mem_by_age"][f"{lo}-{hi}"] = {"ap": avg_precision(pred[m], occ[m]), "base_rate": float(occ[m].mean()),
                                           "iou": float(tp / max(tp + fp + fn, 1)),
                                           "recall": float(tp / max(tp + fn, 1)), "cells": int((occ & m).sum())}
    return res


# ---------------------------------------------------------------- main
name_dirs = dict(s.split("=", 1) for s in args.feats)
eps = load(name_dirs)
tr = [e for e in eps if e["split"] == "train"]
va = [e for e in eps if e["split"] == "val"]
if not va:                                   # tiny debug sets have no val split
    tr, va = tr[:-2], tr[-2:]
print(f"{len(tr)} train / {len(va)} val episodes", flush=True)
rs = np.random.default_rng(0).permutation(len(tr))
ho = [tr[i] for i in rs[:max(len(tr) // 10, 1)]]          # held-out training episodes for early stopping
tr = [tr[i] for i in rs[max(len(tr) // 10, 1):]]
Ytr, Yho = stack(tr, "occ", args.stride), stack(ho, "occ", 4)
lab = {k: stack(va, k) for k in ("occ", "seen", "vis", "age")}
results, preds = {}, {}
sets = list(name_dirs) + ([] if args.no_frame else ["frame"])
for name in sets:
    print("probe:", name, flush=True)
    if name == "frame":
        model = fit(FrameProbe(), stack(tr, "depth", args.stride), Ytr, args.epochs, stack(ho, "depth", 4), Yho)
        pr = predict(model, stack(va, "depth"))
    else:
        Xtr = stack(tr, name, args.stride)
        model = fit(MLPProbe(Xtr.shape[1]), Xtr, Ytr, args.epochs, stack(ho, name, 4), Yho)
        pr = predict(model, stack(va, name))
    results[name] = scores(pr, lab["occ"], lab["seen"], lab["vis"], lab["age"])
    preds[name] = pr
    print(json.dumps({name: {g: results[name][g] for g in ("vis", "mem", "unseen")}}), flush=True)
json.dump(results, open(os.path.join(args.out, "probe_results.json"), "w"), indent=1)

# ---------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name in sets:
    r = results[name]
    k = sets.index(name)
    ax[0].bar(np.arange(3) + 0.25 * k, [r[g]["ap"] for g in ("vis", "mem", "unseen")], 0.25, label=name)
    bins = list(r["mem_by_age"])
    ax[1].plot(range(len(bins)), [r["mem_by_age"][b]["ap"] for b in bins], marker="o", label=name)
ax[0].set_xticks(np.arange(3) + 0.25); ax[0].set_xticklabels(["visible now", "memory", "never seen"])
ax[0].set_ylabel("obstacle average precision"); ax[0].set_title("probe AP by cell group"); ax[0].legend()
ax[1].set_xticks(range(len(bins))); ax[1].set_xticklabels(bins)
ax[1].set_xlabel("steps since last visible (20 Hz)"); ax[1].set_ylabel("obstacle AP (memory cells)")
ax[1].set_title("memory decay"); ax[1].legend(); ax[1].set_ylim(0, 1)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "probe_summary.png"), dpi=130)

# example maps from one val trap episode
ex = next((e for e in va if "_trap_" in e["name"]), va[0])
off = sum(len(e["occ"]) for e in va[:va.index(ex)])
ts = np.linspace(20, len(ex["occ"]) - 1, 5).astype(int)
fig, ax = plt.subplots(len(sets) + 1, len(ts), figsize=(2.2 * len(ts), 2.2 * (len(sets) + 1)))
for j, t in enumerate(ts):
    o, s, v = ex["occ"][t], ex["seen"][t], ex["vis"][t]
    img = np.zeros((64, 64, 3)); img[s & ~o] = 0.45; img[v & ~o] = (0.15, 0.3, 0.55)
    img[o & ~s] = (0.3, 0.08, 0.08); img[o & s & ~v] = (1, 0.6, 0); img[o & v] = (0.9, 0.1, 0.1)
    ax[0, j].imshow(img); ax[0, j].set_title(f"truth t={t / 20:.1f}s", fontsize=8)
    for i, name in enumerate(sets):
        ax[i + 1, j].imshow(preds[name][off + t], vmin=0, vmax=1, cmap="magma")
        ax[i + 1, j].set_title(name, fontsize=8)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(os.path.join(args.out, "probe_examples.png"), dpi=130)

# open-loop imagination error vs horizon (first feature set, val episodes; DreamerV3 features only)
first = list(name_dirs.values())[0]
if "img_depth" not in np.load(os.path.join(first, va[0]["name"])).files:
    print("PROBE_DONE (no img_depth in features, open-loop plot skipped)", flush=True)
    raise SystemExit
errs = {}
for e in va:
    f = np.load(os.path.join(first, e["name"]))
    T = int(f["chunk"]); img = f["img_depth"].astype(np.float32); rec = f["recon_depth"].astype(np.float32)
    true = (e["depth"].astype(np.float32) - sg.D_MIN) / (sg.D_MAX - sg.D_MIN)
    for t in range(len(true)):
        k = t % T - T // 2 + 1
        if k >= 1 and not np.isnan(img[t]).any():
            errs.setdefault(k, []).append(np.abs(img[t] - true[t]).mean() * (sg.D_MAX - sg.D_MIN))
    errs.setdefault(0, []).extend((np.abs(rec - true).mean((1, 2)) * (sg.D_MAX - sg.D_MIN)).tolist())
ks = sorted(errs)
mae = [float(np.mean(errs[k])) for k in ks]
json.dump({"horizon_steps": ks, "depth_mae_m": mae}, open(os.path.join(args.out, "openloop.json"), "w"))
plt.figure(figsize=(5, 3.5)); plt.plot(ks, mae, marker=".")
plt.xlabel("imagination horizon (steps, 0 = posterior reconstruction)"); plt.ylabel("depth MAE (m)")
plt.title("open-loop depth prediction error"); plt.tight_layout()
plt.savefig(os.path.join(args.out, "openloop_error.png"), dpi=130)
print("PROBE_DONE", flush=True)
