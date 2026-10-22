"""Collapse diagnostics of the frame embedding (CLS token) of a TokenJEPA / BeliefJEPA run (main env, from lewm/):
    python percept_diag.py --run_dir $WM_ROOT/runs/kin_v1/m2s --data $WM_ROOT/data/kin_v1 --memtest $WM_ROOT/data/memtest_v1

  rank         RankMe (exp entropy of normalised singular values) on frames of held-out episodes
  top_share    variance share of the top principal direction
  sigreg_N     SIGReg statistic of those frames estimated from N samples (training used N = batch, or
               loss.sigreg.samples); i.i.d. N(0, I) scores ~1.0 at every N
  wall_signal  mean ||z(W in view) - z(no W)||^2 on the hover frames of the memtest seen / absent twins
Writes <run_dir>/percept_diag.json.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

from jepa import encode_tokens
from module import SIGReg
from train import effective_rank
from utils import depth_to_pixels, episode_files, load_ckpt, D_MIN, D_MAX

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--data", required=True)
p.add_argument("--memtest", required=True)
p.add_argument("--episodes", type=int, default=60)
args = p.parse_args()
torch.manual_seed(0)

ckpt = load_ckpt(Path(args.run_dir) / "lewm_object.ckpt")
model, k = ckpt["model"].cuda().eval(), ckpt["cfg"]["data"]["frameskip"]


@torch.no_grad()
def cls(depth, stride=1):
    fr = np.arange(0, len(depth), k)[::stride]
    px = np.round((np.clip(depth[fr].astype(np.float32), D_MIN, D_MAX) - D_MIN) / (D_MAX - D_MIN) * 255)
    z = encode_tokens(model.encoder, model.projector, depth_to_pixels(px.astype(np.uint8))[None].cuda(), model.spatial)
    return z[0][:, 0].float()  # (F, D)


Z = torch.cat([cls(np.load(f)["depth"], stride=4) for f in episode_files(args.data, "val")[:args.episodes]])
Z = Z[torch.randperm(len(Z))]
sv = torch.linalg.svdvals(Z - Z.mean(0))
sigreg = SIGReg().cuda()
out = {"frames": len(Z), "rank": effective_rank(Z).item(), "top_share": (sv[0] ** 2 / (sv ** 2).sum()).item()}
for N in (16, 128):
    with torch.no_grad():
        out[f"sigreg_{N}"] = float(np.mean([sigreg(Z[i * N:(i + 1) * N][None]).item() for i in range(min(20, len(Z) // N))]))

sig = []
for f in sorted(glob.glob(os.path.join(args.memtest, "sc_*_gap000_seen.npz"))):
    seen, absent = np.load(f), np.load(f.replace("_seen", "_absent"))
    fr = np.arange(0, len(seen["depth"]), k)
    pre = (seen["phase"][fr] < 2) & (seen["wall_px"][fr] > 200)  # hover phase with W in view
    if pre.any():
        sig.append((cls(seen["depth"])[pre] - cls(absent["depth"])[pre]).pow(2).mean().item())
out["wall_signal"] = float(np.mean(sig))
print(json.dumps({kk: round(v, 4) for kk, v in out.items()}), flush=True)
json.dump(out, open(Path(args.run_dir) / "percept_diag.json", "w"), indent=1)
print("PERCEPT_DIAG_DONE", flush=True)
