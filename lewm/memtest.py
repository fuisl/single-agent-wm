"""Label-free memory test on the twin look-away scenarios of uav_env/kin_memtest.py (main env, run from lewm/):
    python memtest.py --run_dir $WM_ROOT/runs/kin_v1/m2 --data $WM_ROOT/data/memtest_v1

Every rollout is seen as frames every frameskip steps from step 0. For each frame j the model predicts its
embedding from the past only: window models (official LeWM, TokenJEPA) run the predictor on the history_size
frames before j, belief models (BeliefJEPA) the one-step open-loop head from b_{j-1}. The error e_j is the squared
latent prediction error divided by the variance of the target embedding over all seen-variant frames.
Re-entry frames: the first 4 frames in the turn-back / hover phases where the wall W is visible.

  E1 memory effect  mean over scenarios of e(novel) - e(seen) on re-entry frames: identical frames, so a positive
                    value can only come from having seen W before the gap
  E2 surprise       e(removed) - e(absent) on the same frames (W gone vs never there: identical frames), and the VoE
                    accuracy P(e(removed) > e(absent)) (paired by scenario)
  E3 lesions        (belief models) seen rollout with b replaced at the start of the turn-back by the novel twin's
                    belief (swap) or by zeros (reset): the share of the E1 effect that the belief carries
Writes <run_dir>/memtest.json and memtest_traces.npz (mean error per frame around the turn-back, per gap/variant).
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from jepa import JEPA, BeliefJEPA, TokenJEPA, encode_tokens
from utils import ZScore, action_blocks, depth_to_pixels, load_ckpt, D_MIN, D_MAX

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--data", required=True)
p.add_argument("--reentry_frames", type=int, default=4)
p.add_argument("--min_wall_px", type=int, default=200)
args = p.parse_args()
run_dir = Path(args.run_dir)

ckpt = load_ckpt(run_dir / "lewm_object.ckpt")
model, cfg, stats = ckpt["model"], ckpt["cfg"], ckpt["stats"]
kind = "belief" if isinstance(model, BeliefJEPA) else "token" if isinstance(model, TokenJEPA) else "lewm"
k, HS = cfg["data"]["frameskip"], cfg["history_size"]
snorm = ZScore(*[s.cuda() for s in stats["state"]])
anorm = ZScore(*[s.cuda() for s in stats["action"]])


def load(f):
    d = np.load(f)
    depth = np.round((d["depth"].astype(np.float32) - D_MIN) / (D_MAX - D_MIN) * 255).astype(np.uint8)
    fr = np.arange(0, len(depth), k)
    return {"depth": depth[fr], "state": d["state"][fr], "blocks": action_blocks(d["action"], fr, k),
            "phase": d["phase"][fr], "wall_px": d["wall_px"][fr]}


@torch.no_grad()
def embed(r):
    px = depth_to_pixels(r["depth"])[None].cuda()
    s = snorm(torch.as_tensor(r["state"], device="cuda"))
    a = anorm(torch.as_tensor(r["blocks"], device="cuda"))
    if kind == "lewm":
        z = model.encode({"pixels": px, "state": s[None]})["emb"][0][:, None]  # (F, 1, D)
    else:
        z = encode_tokens(model.encoder, model.projector, px, model.spatial)[0]  # (F, N, D)
    return z, s, a


@torch.no_grad()
def errors(z, s, a, b0=None, start=0):
    """Squared one-step prediction error per frame (F,), NaN where undefined. Belief models: the belief is run
    from frame `start` with initial state b0 (lesions); returns the beliefs too."""
    F = len(z)
    e = torch.full((F,), float("nan"), device="cuda")
    if kind == "belief":
        a_prev = torch.cat([torch.zeros_like(a[:1]), a[:-1]])
        if hasattr(model.belief, "warp"):  # MapBelief: lesions act on the map, which is its memory
            b, mem = (x[0] for x in model.belief.observe(z[start:][None], s[start:][None], a_prev[start:][None],
                                                         b0=b0, return_map=True))
        else:
            b = mem = model.belief.observe(z[start:][None], s[start:][None], a_prev[start:][None], b0=b0)[0]
        tok, _, _ = model.belief.imagine(b[:-1], a[start:-1, None])
        e[start + 1:] = (tok[:, 0] - z[start + 1:]).pow(2).mean((-1, -2))
        return e, mem
    for j in range(HS, F):
        ctx = slice(j - HS, j)
        if kind == "lewm":
            pr = model.predict(z[None, ctx, 0], model.action_encoder(a[None, ctx]))[0, -1]
            e[j] = (pr - z[j, 0]).pow(2).mean()
        else:
            pr, _ = model.predict(z[None, ctx], s[None, ctx], model.action_encoder(a[None, ctx]))
            e[j] = (pr[0, -1] - z[j]).pow(2).mean()
    return e, None


files = sorted(glob.glob(os.path.join(args.data, "sc_*_seen.npz")))
assert files, f"no scenarios in {args.data}"
# target variance over all seen-variant frames
zs = [embed(load(f))[0] for f in files[::7]]
var = torch.cat(zs).flatten(0, 1).var(0).mean().item()

res = defaultdict(lambda: defaultdict(list))
traces = defaultdict(list)
for f in files:
    gap = int(np.load(f)["gap"])
    runs = {v: load(f.replace("_seen.npz", f"_{v}.npz")) for v in ("seen", "novel", "removed", "absent")}
    j0 = int(np.argmax(runs["seen"]["phase"] >= 3))  # first turn-back frame
    re = np.nonzero((runs["seen"]["phase"] >= 3) & (runs["seen"]["wall_px"] > args.min_wall_px))[0][:args.reentry_frames]
    out, bel = {}, {}
    for v, r in runs.items():
        z, s, a = embed(r)
        out[v], bel[v] = errors(z, s, a)
        if v == "seen":
            zs_seen = (z, s, a)
    if kind == "belief":
        z, s, a = zs_seen
        out["swap"] = errors(z, s, a, b0=bel["novel"][j0 - 1:j0], start=j0)[0]
        out["reset"] = errors(z, s, a, b0=torch.zeros_like(bel["seen"][:1]), start=j0)[0]
    e = {v: (x / var).cpu().numpy() for v, x in out.items()}
    for v in e:
        res[gap][v].append(float(np.nanmean(e[v][re])))
        traces[(gap, v)].append(e[v][j0 - 6:j0 + 12])

summary = {"kind": kind, "target_var": var, "gaps": {}}
for gap in sorted(res):
    r = {v: np.array(x) for v, x in res[gap].items()}
    d = r["novel"] - r["seen"]
    g = {"n": len(d), "reentry_err_seen": float(r["seen"].mean()), "reentry_err_novel": float(r["novel"].mean()),
         "memory_effect": float(d.mean()), "memory_effect_se": float(d.std(ddof=1) / np.sqrt(len(d))),
         "surprise": float((r["removed"] - r["absent"]).mean()),
         "surprise_se": float((r["removed"] - r["absent"]).std(ddof=1) / np.sqrt(len(d))),
         "voe_acc": float((r["removed"] > r["absent"]).mean())}
    if "swap" in r:
        g["swap_effect"] = float((r["swap"] - r["seen"]).mean())
        g["reset_effect"] = float((r["reset"] - r["seen"]).mean())
        g["belief_share"] = float(g["swap_effect"] / g["memory_effect"]) if abs(g["memory_effect"]) > 1e-9 else None
    summary["gaps"][gap] = g
    print(json.dumps({"gap_steps": gap, **{kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in g.items()}}),
          flush=True)
json.dump(summary, open(run_dir / "memtest.json", "w"), indent=1)
np.savez(run_dir / "memtest_traces.npz", **{f"gap{g:03d}_{v}": np.nanmean(np.stack(x), 0) for (g, v), x in traces.items()})
print("MEMTEST_DONE", flush=True)
