"""Per-step LeWM features for probe/probe.py, plus decoded reconstructions and open-loop imagination
(main env, run from lewm/):
    python extract.py --run_dir $WM_ROOT/runs/lewm --data $WM_ROOT/data/wm_v1
    python extract.py --run_dir $WM_ROOT/runs/lewm --data $WM_ROOT/data/wm_v1 --random_init

Writes <run_dir>/<prefix>_ctx/ and <prefix>_emb/ with one npz per episode (same file names as the data):
  feat: (T, D) after observing step t; ctx = the predictor's history window [e_{t-2k}, e_{t-k}, e_t]
        (all LeWM carries from the past), emb = e_t only
  split, chunk; ctx also has recon_depth (decoded e_t) and img_depth: per chunk of `chunk` steps, observe up
  to the middle and imagine the second half from the recorded actions (one frame every frameskip steps,
  NaN elsewhere), aligned like dreamer/train_wm.py extract so probe.py's open-loop plot works unchanged.
"""
import argparse
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from utils import ZScore, action_blocks, encode_steps, episode_files, is_val, load_ckpt, load_episode

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--data", required=True)
p.add_argument("--chunk", type=int, default=64)
p.add_argument("--max_extract_train", type=int, default=600, help="train episodes to extract for probe fitting")
p.add_argument("--random_init", action="store_true", help="untrained LeWM (same architecture) as a baseline")
args = p.parse_args()
run_dir = Path(args.run_dir)

ckpt = load_ckpt(run_dir / "lewm_object.ckpt")
cfg, stats = ckpt["cfg"], ckpt["stats"]
if args.random_init:
    torch.manual_seed(0)
    ckpt["model"] = hydra.utils.instantiate(OmegaConf.create(cfg).model).cuda().eval().requires_grad_(False)
model = ckpt["model"]
heads = None if args.random_init else torch.load(run_dir / "heads.pt", weights_only=False)
k, HS = cfg["data"]["frameskip"], cfg["history_size"]
act_norm = ZScore(*[s.cuda() for s in stats["action"]])
prefix = "features_rand" if args.random_init else "features"
out_ctx, out_emb = run_dir / f"{prefix}_ctx", run_dir / f"{prefix}_emb"
out_ctx.mkdir(exist_ok=True); out_emb.mkdir(exist_ok=True)


@torch.no_grad()
def decode(x):
    return torch.cat([heads["decoder"](x[i:i + 1024].float()) for i in range(0, len(x), 1024)]).half().cpu().numpy()


@torch.no_grad()
def imagine(emb, action, T):
    """Open-loop: for each chunk, history ends at its middle step t0, then imagine to the chunk end."""
    img = np.full((T, 48, 80), np.nan, np.float16)
    P = (args.chunk - args.chunk // 2) // k
    for c in range(0, T, args.chunk):
        t0 = c + args.chunk // 2 - 1
        hist = [t0 - j * k for j in range(HS - 1, -1, -1) if t0 - j * k >= 0]
        fut = [t0 + i * k for i in range(1, P + 1) if t0 + i * k < T]
        if t0 >= T or not fut:
            continue
        blocks = action_blocks(action, hist + [t0 + i * k for i in range(1, len(fut))], k)
        pred = model.rollout(emb[hist][None], act_norm(torch.as_tensor(blocks, device="cuda"))[None], len(fut), HS)
        img[fut] = decode(pred[0])
    return img


rng = np.random.default_rng(1)
chosen = episode_files(args.data, "val") + list(rng.permutation(episode_files(args.data, "train"))[:args.max_extract_train])
for n, f in enumerate(chosen):
    e = load_episode(f)
    T = len(e["reward"])
    emb = encode_steps(model, stats, e["depth"], e["state"])
    idx = np.maximum(np.arange(T)[:, None] - k * np.arange(HS - 1, -1, -1), 0)  # (T, HS), clamped at the start
    ctx = emb[torch.as_tensor(idx, device=emb.device)].flatten(1)
    split = "val" if is_val(f) else "train"
    name = os.path.basename(f)
    np.savez(out_emb / name, feat=emb.half().cpu().numpy(), split=split, chunk=args.chunk)
    extra = {} if heads is None else {"recon_depth": decode(emb), "img_depth": imagine(emb, e["action"], T)}
    np.savez(out_ctx / name, feat=ctx.half().cpu().numpy(), split=split, chunk=args.chunk, **extra)
    if (n + 1) % 50 == 0 or n + 1 == len(chosen):
        print(f"extracted {n + 1}/{len(chosen)}", flush=True)
print("EXTRACT_DONE", flush=True)
