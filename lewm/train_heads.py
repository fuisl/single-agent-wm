"""Post-hoc readouts on the frozen LeWM encoder (main env, run from lewm/):
    python train_heads.py --run_dir $WM_ROOT/runs/lewm --data $WM_ROOT/data/wm_v1
  * DepthDecoder: emb -> depth, for visualising reconstructions and imagined rollouts (as in the LeWM paper)
  * RewardHead:   emb -> (reward, terminal) used by the CEM planner in eval.py
The world model gets no gradient from these. Saves <run_dir>/heads.pt.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from module import DepthDecoder, RewardHead
from utils import encode_steps, episode_files, load_ckpt, load_episode

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--data", required=True)
p.add_argument("--steps", type=int, default=10000)
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--term_pos_weight", type=float, default=20.0)
args = p.parse_args()
run_dir = Path(args.run_dir)

ckpt = load_ckpt(run_dir / "lewm_object.ckpt")
model, stats = ckpt["model"], ckpt["stats"]
emb, depth, reward, term = [], [], [], []
for f in episode_files(args.data, "train"):
    e = load_episode(f)
    emb.append(encode_steps(model, stats, e["depth"], e["state"]).float().cpu())
    depth.append(torch.from_numpy(e["depth"]))
    reward.append(torch.from_numpy(e["reward"]))
    term.append(torch.from_numpy(e["is_terminal"]).float())
emb, depth, reward, term = map(torch.cat, (emb, depth, reward, term))
print(f"{len(emb)} steps, {int(term.sum())} terminal", flush=True)

D = emb.shape[1]
decoder, head = DepthDecoder(D).cuda(), RewardHead(D).cuda()
opt = torch.optim.AdamW([*decoder.parameters(), *head.parameters()], lr=3e-4, weight_decay=1e-4)
pos_w = torch.tensor(args.term_pos_weight, device="cuda")
log = open(run_dir / "heads_metrics.jsonl", "w")
t0 = time.time()
for step in range(1, args.steps + 1):
    idx = torch.randint(len(emb), (args.batch_size,))
    x = emb[idx].cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rec = decoder(x).float()
        r, t_logit = (y.float() for y in head(x))
    losses = {
        "depth_l1": (rec - depth[idx].cuda().float() / 255).abs().mean(),
        "reward_mse": F.mse_loss(r, reward[idx].cuda()),
        "term_bce": F.binary_cross_entropy_with_logits(t_logit, term[idx].cuda(), pos_weight=pos_w),
    }
    opt.zero_grad(set_to_none=True)
    sum(losses.values()).backward()
    opt.step()
    if step % 500 == 0:
        m = {k: round(v.item(), 4) for k, v in losses.items()} | {"step": step, "minutes": round((time.time() - t0) / 60, 1)}
        log.write(json.dumps(m) + "\n"); log.flush()
        print(json.dumps(m), flush=True)
torch.save({"decoder": decoder.eval(), "head": head.eval()}, run_dir / "heads.pt")
print("HEADS_DONE", flush=True)
