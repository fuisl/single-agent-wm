"""Per-step features of the R-LeWM variants (TokenJEPA M0/M1, BeliefJEPA M2/M3) for probe/probe.py
(main env, run from lewm/):
    python extract_mem.py --run_dir $WM_ROOT/runs/kin_v1/m2 --data $WM_ROOT/data/kin_v1

Writes one npz per episode (feat (T, F), split, chunk) to
  <run_dir>/features_percept  frame tokens z_t (N * D): what the encoder sees now
  <run_dir>/features_pred     TokenJEPA: predicted next-frame tokens from the window ending at t (the predictor's
                              view of the past, 3 frames)
  <run_dir>/features_belief   BeliefJEPA: belief b_t after observing step t (frames every frameskip steps ending at t)
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from jepa import BeliefJEPA, encode_tokens
from utils import ZScore, action_blocks, depth_to_pixels, episode_files, is_val, load_ckpt, load_episode

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--data", required=True)
p.add_argument("--max_extract_train", type=int, default=300)
args = p.parse_args()
run_dir = Path(args.run_dir)

ckpt = load_ckpt(run_dir / "lewm_object.ckpt")
model, cfg, stats = ckpt["model"], ckpt["cfg"], ckpt["stats"]
belief = isinstance(model, BeliefJEPA)
k, HS = cfg["data"]["frameskip"], cfg["history_size"]
snorm = ZScore(*[s.cuda() for s in stats["state"]])
anorm = ZScore(*[s.cuda() for s in stats["action"]])
outs = ["percept", "belief" if belief else "pred"]
for o in outs:
    (run_dir / f"features_{o}").mkdir(exist_ok=True)


@torch.no_grad()
def episode_features(e):
    T = len(e["reward"])
    z = torch.cat([encode_tokens(model.encoder, model.projector, depth_to_pixels(e["depth"][c:c + 512])[None].cuda(),
                                 model.spatial)[0] for c in range(0, T, 512)])  # (T, N, D)
    s = snorm(torch.as_tensor(e["state"], device="cuda"))
    blocks = anorm(torch.as_tensor(action_blocks(e["action"], np.arange(T), k), device="cuda"))  # block after step t
    if belief:
        feat = torch.zeros(T, model.belief.belief_dim, device="cuda")
        for ph in range(k):  # frames ph, ph + k, ...: one belief sequence per phase covers every step
            st = torch.arange(ph, T, k, device="cuda")
            a_prev = torch.cat([torch.zeros_like(blocks[:1]), blocks[st[:-1]]])
            feat[st] = model.belief.observe(z[st][None], s[st][None], a_prev[None])[0]
    else:
        idx = torch.as_tensor(np.maximum(np.arange(T)[:, None] - k * np.arange(HS - 1, -1, -1), 0), device="cuda")
        feat = []
        for c in range(0, T, 256):
            i = idx[c:c + 256]
            pe, _ = model.predict(z[i], s[i], model.action_encoder(blocks[i]))
            feat.append(pe[:, -1].flatten(1))
        feat = torch.cat(feat)
    return {"percept": z.flatten(1), outs[1]: feat}


rng = np.random.default_rng(1)
chosen = episode_files(args.data, "val") + list(rng.permutation(episode_files(args.data, "train"))[:args.max_extract_train])
for n, f in enumerate(chosen):
    feats = episode_features(load_episode(f))
    split = "val" if is_val(f) else "train"
    for o, v in feats.items():
        np.savez(run_dir / f"features_{o}" / os.path.basename(f), feat=v.half().cpu().numpy(), split=split, chunk=64)
    if (n + 1) % 50 == 0 or n + 1 == len(chosen):
        print(f"extracted {n + 1}/{len(chosen)}", flush=True)
print("EXTRACT_DONE", flush=True)
