"""Post-hoc readouts on the frozen LeWM encoder (main env, run from lewm/):
    python train_heads.py --run_dir $WM_ROOT/runs/lewm --data $WM_ROOT/data/wm_v1
  * DepthDecoder: emb -> depth, for visualising reconstructions and imagined rollouts (as in the LeWM paper)
  * RewardHead:   emb -> (reward, terminal) used by the CEM planner in eval.py
The world model gets no gradient from these. Saves <run_dir>/heads.pt.
--rep cls / patch trains only a depth decoder on the ViT CLS (before the state is added) or on the 6x10 patch-token
grid, saved as <run_dir>/heads_<rep>.pt, to see what each encoder stage still holds.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from module import DepthDecoder, PatchDecoder, RewardHead
from utils import depth_to_pixels, encode_steps, episode_files, load_ckpt, load_episode

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--data", required=True)
p.add_argument("--steps", type=int, default=10000)
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--term_pos_weight", type=float, default=20.0)
p.add_argument("--rep", default="emb", choices=["emb", "cls", "patch"])
p.add_argument("--max_episodes", type=int, default=None, help="train episodes used (RAM: patch grids are 11.5k-D)")
p.add_argument("--stride", type=int, default=1)
args = p.parse_args()
run_dir = Path(args.run_dir)

ckpt = load_ckpt(run_dir / "lewm_object.ckpt")
model, stats = ckpt["model"], ckpt["stats"]


@torch.no_grad()
def features(e):
    if args.rep == "emb":
        return encode_steps(model, stats, e["depth"], e["state"]).float()
    out = []
    for c in range(0, len(e["depth"]), 512):
        hs = model.encoder(depth_to_pixels(e["depth"][c:c + 512]).cuda()).last_hidden_state
        out.append(hs[:, 0] if args.rep == "cls" else hs[:, 1:].reshape(len(hs), 6, 10, -1).permute(0, 3, 1, 2))
    return torch.cat(out)


emb, depth, reward, term = [], [], [], []
files = episode_files(args.data, "train")[:args.max_episodes]
for f in files:
    e = load_episode(f)
    s = slice(None, None, args.stride)
    emb.append(features(e)[s].half().cpu())
    depth.append(torch.from_numpy(e["depth"][s]))
    reward.append(torch.from_numpy(e["reward"][s]))
    term.append(torch.from_numpy(e["is_terminal"][s]).float())
emb, depth, reward, term = map(torch.cat, (emb, depth, reward, term))
print(f"{len(emb)} steps, {int(term.sum())} terminal, rep {args.rep} {tuple(emb.shape[1:])}", flush=True)

D = emb.shape[1]
decoder = (PatchDecoder(D) if args.rep == "patch" else DepthDecoder(D)).cuda()
head = RewardHead(D).cuda() if args.rep == "emb" else None
params = [*decoder.parameters(), *(head.parameters() if head else [])]
opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
pos_w = torch.tensor(args.term_pos_weight, device="cuda")
log = open(run_dir / ("heads_metrics.jsonl" if args.rep == "emb" else f"heads_{args.rep}_metrics.jsonl"), "w")
t0 = time.time()
for step in range(1, args.steps + 1):
    idx = torch.randint(len(emb), (args.batch_size,))
    x = emb[idx].cuda().float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rec = decoder(x).float()
        if head:
            r, t_logit = (y.float() for y in head(x))
    losses = {"depth_l1": (rec - depth[idx].cuda().float() / 255).abs().mean()}
    if head:
        losses["reward_mse"] = F.mse_loss(r, reward[idx].cuda())
        losses["term_bce"] = F.binary_cross_entropy_with_logits(t_logit, term[idx].cuda(), pos_weight=pos_w)
    opt.zero_grad(set_to_none=True)
    sum(losses.values()).backward()
    opt.step()
    if step % 500 == 0:
        m = {k: round(v.item(), 4) for k, v in losses.items()} | {"step": step, "minutes": round((time.time() - t0) / 60, 1)}
        log.write(json.dumps(m) + "\n"); log.flush()
        print(json.dumps(m), flush=True)
if head:
    torch.save({"decoder": decoder.eval(), "head": head.eval()}, run_dir / "heads.pt")
else:
    torch.save({"decoder": decoder.eval()}, run_dir / f"heads_{args.rep}.pt")
print("HEADS_DONE", flush=True)
