"""Does the belief's history help prediction on ordinary flight? (main env, run from lewm/):
    python reset_check.py --run_dir $WM_ROOT/runs/kin_v1/m2s --data $WM_ROOT/data/kin_v1 [--device cpu]

On held-out episodes (frames every frameskip steps), the belief is set to zeros before frame r and the one-step
latent error on the next `horizon` frames is compared with an intact belief (memtest.py's reset lesion, off the
memtest scenarios). Two references for the intact belief:
  window  started from zeros `history` frames before r, as in training (32-frame windows from a zero belief)
  full    run from the start of the episode (longer than any training window)
effect = (e_reset - e_intact) / target variance, mean ± s.e. over resets; > 0: the history helps.
Writes <run_dir>/reset_check.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jepa import encode_tokens
from utils import ZScore, action_blocks, depth_to_pixels, episode_files, load_ckpt, load_episode

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--data", required=True)
p.add_argument("--episodes", type=int, default=60)
p.add_argument("--history", type=int, default=16)
p.add_argument("--horizon", type=int, default=16)
p.add_argument("--device", default="cuda")
args = p.parse_args()
dev = args.device

ckpt = load_ckpt(Path(args.run_dir) / "lewm_object.ckpt", device=dev)
model, cfg, stats = ckpt["model"], ckpt["cfg"], ckpt["stats"]
k = cfg["data"]["frameskip"]
snorm, anorm = ZScore(*stats["state"]), ZScore(*stats["action"])


@torch.no_grad()
def episode(e):
    fr = np.arange(0, len(e["reward"]), k)
    z = encode_tokens(model.encoder, model.projector, depth_to_pixels(e["depth"][fr])[None].to(dev), model.spatial)[0]
    s = snorm(torch.as_tensor(e["state"][fr], device=dev))
    a = anorm(torch.as_tensor(action_blocks(e["action"], fr, k), device=dev))
    a_prev = torch.cat([torch.zeros_like(a[:1]), a[:-1]])
    F, H = len(z), args.history
    rs = list(range(H, F - args.horizon - 1, args.horizon))
    # rows: full intact, then per r: full reset at r, window intact (zeros at r - H), window reset (also at r)
    reset = torch.zeros(1 + 3 * len(rs), F, dtype=torch.bool, device=dev)
    for i, r in enumerate(rs):
        reset[1 + 3 * i, r] = True
        reset[2 + 3 * i, r - H] = True
        reset[3 + 3 * i, [r - H, r]] = True
    n = len(reset)
    b = model.belief.observe(z[None].expand(n, *z.shape), s[None].expand(n, -1, -1), a_prev[None].expand(n, -1, -1),
                             reset=reset)  # (n, F, Bd)
    tok, _, _ = model.belief.imagine(b[:, :-1].flatten(0, 1), a[None, :-1, None].expand(n, -1, -1, -1).flatten(0, 1))
    e = (tok[:, 0].view(n, F - 1, *z.shape[1:]) - z[None, 1:]).pow(2).mean((-1, -2))  # e[:, t]: error of frame t + 1
    out = {"full": [], "window": [], "intact_full": [], "intact_window": []}
    for i, r in enumerate(rs):
        seg = slice(r, r + args.horizon)  # frames r + 1 .. r + horizon, predicted from b_r ..
        out["full"].append((e[1 + 3 * i, seg] - e[0, seg]).cpu().numpy())
        out["window"].append((e[3 + 3 * i, seg] - e[2 + 3 * i, seg]).cpu().numpy())
        out["intact_full"].append(e[0, seg].cpu().numpy())
        out["intact_window"].append(e[2 + 3 * i, seg].cpu().numpy())
    return out, z


res, zs = {}, []
for f in episode_files(args.data, "val")[:args.episodes]:
    o, z = episode(load_episode(f))
    zs.append(z[::4])
    for key, v in o.items():
        res.setdefault(key, []).extend(v)
var = torch.cat(zs).flatten(0, 1).var(0).mean().item()
summary = {"target_var": var, "resets": len(res["full"]), "history": args.history}
for ref in ("window", "full"):
    d = np.stack(res[ref]) / var  # (resets, horizon)
    base = np.stack(res[f"intact_{ref}"]).mean() / var
    summary[ref] = {"effect": d.mean(0).tolist(), "effect_se": (d.std(0, ddof=1) / np.sqrt(len(d))).tolist(),
                    "mean_effect": float(d.mean()), "mean_effect_se": float(d.mean(1).std(ddof=1) / np.sqrt(len(d))),
                    "intact_err": float(base)}
    print(f"{ref:6s} intact err {base:.4f}  reset effect {d.mean():+.4f} ± {summary[ref]['mean_effect_se']:.4f} "
          f"({100 * d.mean() / base:+.1f} %)  by frame after reset: "
          + " ".join(f"{x:+.3f}" for x in d.mean(0)[:8]), flush=True)
json.dump(summary, open(Path(args.run_dir) / "reset_check.json", "w"), indent=1)
print("RESET_CHECK_DONE", flush=True)
