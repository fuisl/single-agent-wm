"""Where does the obstacle content of a future frame come from?  (data only, CPU)
For occupied cells visible at t+k, classify relative to what LeWM's predictor sees at time t (frames t-10, t-5, t):
  ctx    = visible in one of the 3 context frames
  gapK   = not in the context frames but seen before (needs memory), binned by steps since last visible
  novel  = never seen up to t (nothing can predict it except priors)
Usage (main env, after source env.sh): python scripts/diag/reobs.py <data_dir> <n_episodes>
"""
import glob
import sys

import numpy as np

import scenegen as sg

data, n_eps = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(f"{data}/ep_*.npz"))[:n_eps]
X, Y = np.meshgrid(*(2 * [-sg.EXTENT + (np.arange(sg.SEEN_N) + 0.5) * sg.SEEN_RES]), indexing="ij")
PTS = np.stack([X, Y], -1)
w = int(np.ceil((sg.D_MAX + 0.3) / sg.SEEN_RES))
BINS = [(1, 15), (16, 64), (65, 10 ** 9)]
KS = [5, 25]  # next LeWM frame (0.25 s), end of the CEM horizon (1.25 s)
tot = {k: {"ctx": 0, "novel": 0, **{f"gap{a}-{b}": 0 for a, b in BINS}} for k in KS}
steps_with_mem = {k: [0, 0] for k in KS}
for f in files:
    d = np.load(f)
    boxes = d["boxes"].astype(np.float64)
    occ = sg.min_dist(PTS.reshape(-1, 2), boxes).reshape(sg.SEEN_N, sg.SEEN_N) <= 0
    pos, yaw = d["pos"].astype(np.float64), d["yaw"].astype(np.float64)
    T = len(pos)
    vis = np.zeros((T, sg.SEEN_N, sg.SEEN_N), bool)
    for t in range(T):
        dirs = np.stack([np.cos(yaw[t] + sg.RAY_ANG), np.sin(yaw[t] + sg.RAY_ANG)], -1)
        rr = sg.raycast2d(pos[t, :2], dirs, boxes, sg.D_MAX + 1.0)
        ci, cj = ((pos[t, :2] + sg.EXTENT) / sg.SEEN_RES).astype(int)
        i0, i1, j0, j1 = max(ci - w, 0), min(ci + w + 1, sg.SEEN_N), max(cj - w, 0), min(cj + w + 1, sg.SEEN_N)
        if i0 < i1 and j0 < j1:
            vis[t, i0:i1, j0:j1] = sg.visibility(PTS[i0:i1, j0:j1], pos[t], yaw[t], rr)
    vis &= occ
    last = np.full((sg.SEEN_N, sg.SEEN_N), -10 ** 9)
    for t in range(T):
        last[vis[t]] = t
        if t < 10 or t % 5:
            continue
        ctx = vis[t] | vis[t - 5] | vis[t - 10]
        for k in KS:
            if t + k >= T:
                continue
            cells = vis[t + k]
            tot[k]["ctx"] += int((cells & ctx).sum())
            rest = cells & ~ctx
            age = t - last[rest]
            tot[k]["novel"] += int((age > 10 ** 8).sum())
            for a, b in BINS:
                tot[k][f"gap{a}-{b}"] += int(((age >= a) & (age <= b)).sum())
            steps_with_mem[k][0] += int((age <= 10 ** 8).any())
            steps_with_mem[k][1] += 1
for k in KS:
    n = sum(tot[k].values())
    print(f"k=+{k} steps: obstacle cells visible at t+k = {n}; " +
          ", ".join(f"{c} {v / n:.3f}" for c, v in tot[k].items()) +
          f"; frames needing memory {steps_with_mem[k][0] / steps_with_mem[k][1]:.3f}")
