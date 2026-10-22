"""Control features for the belief probe: the belief's inputs other than the image (main env, run from lewm/):
    python extract_state.py --run_dir $WM_ROOT/runs/kin_v1/m2s --data $WM_ROOT/data/kin_v1

Writes <run_dir>/features_state/ep_*.npz (feat (T, F), split, chunk) for the episodes of <run_dir>/features_belief:
the z-scored state and previous action block at the last `frames` frames (every frameskip steps, ending at t;
repeated first frame before the episode start), i.e. what a belief over the training window could integrate
without ever seeing a depth image. A probe on these bounds how much of the belief's memory AP the state can give.
"""
import argparse
import os
from pathlib import Path

import numpy as np

from utils import action_blocks, load_ckpt, load_episode

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--data", required=True)
p.add_argument("--frames", type=int, default=32)
args = p.parse_args()
run_dir = Path(args.run_dir)

ckpt = load_ckpt(run_dir / "lewm_object.ckpt", device="cpu")
k, stats = ckpt["cfg"]["data"]["frameskip"], ckpt["stats"]
sm, ss = (x.cpu().numpy() for x in stats["state"])
am, as_ = (x.cpu().numpy() for x in stats["action"])
out = run_dir / "features_state"
out.mkdir(exist_ok=True)

ref = sorted((run_dir / "features_belief").glob("ep_*.npz"))
assert ref, f"no features_belief in {run_dir}"
for n, r in enumerate(ref):
    e, meta = load_episode(os.path.join(args.data, r.name)), np.load(r)
    T = len(e["reward"])
    s = (e["state"] - sm) / ss
    a_prev = (action_blocks(e["action"], np.maximum(np.arange(T) - k, 0), k) - am) / as_
    a_prev[:k] = 0  # no action before the first frame, as in the belief
    idx = np.maximum(np.arange(T)[:, None] - k * np.arange(args.frames), 0)  # (T, frames), newest first
    feat = np.concatenate([s[idx], a_prev[idx]], -1).reshape(T, -1)
    np.savez(out / r.name, feat=feat.astype(np.float16), split=meta["split"], chunk=meta["chunk"])
    if (n + 1) % 50 == 0 or n + 1 == len(ref):
        print(f"extracted {n + 1}/{len(ref)}", flush=True)
print("EXTRACT_DONE", flush=True)
