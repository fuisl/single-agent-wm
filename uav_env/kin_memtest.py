"""Twin look-away scenarios for a label-free memory test (docs/lewm_memory_review.md §5.2, E1-E3).

    python kin_memtest.py --out data/memtest_v1 --scenarios 60 --workers 12

The drone hovers facing a wall W (3 s), turns away by 110-150 deg (W leaves the 87 deg view), holds for a gap,
turns back (W re-enters) and hovers. Every scenario is rendered for each gap with three variants of W:
  seen     W present throughout
  novel    W absent before the gap, inserted while the drone looks away: identical frames from the gap on, but
           nothing to remember
  removed  W present before the gap, removed while the drone looks away: a violation of expectation at re-entry
  absent   W never present: identical frames to "removed" from the gap on, but nothing to expect
Other boxes (pillars, away from the line of sight to W) are present in every variant. Per step it stores depth,
state, action, pos, yaw, the phase (0 face, 1 turn away, 2 gap, 3 turn back, 4 hover) and wall_px: the number of
depth pixels that W occupies in the seen variant (re-entry frames are phase >= 3 with wall_px > 0).
"""
import argparse
import multiprocessing as mp
import os

import numpy as np

import scenegen as sg
from kinsim import KinDrone

p = argparse.ArgumentParser()
p.add_argument("--out", default="data/memtest_v1")
p.add_argument("--scenarios", type=int, default=60)
p.add_argument("--gaps", default="0,10,20,40,80,160", help="hold time with W out of view, env steps (20 Hz)")
p.add_argument("--workers", type=int, default=12)
p.add_argument("--seed", type=int, default=7)
args = p.parse_args()
GAPS = [int(g) for g in args.gaps.split(",")]
FACE, TURN, HOVER = 60, 30, 20


def scenario(rng):
    yaw0 = rng.uniform(-np.pi, np.pi)
    fwd = np.array([np.cos(yaw0), np.sin(yaw0)])
    d, half = rng.uniform(1.8, 3.0), rng.uniform(0.8, 1.5)
    wall = np.array([*(d * fwd), half, 0.15, yaw0 + np.pi / 2])
    side, theta = rng.choice([-1, 1]), rng.uniform(np.deg2rad(110), np.deg2rad(150))
    pillars = []
    while len(pillars) < rng.integers(4, 9):
        r, a = rng.uniform(2.0, 6.0), yaw0 + rng.uniform(-np.pi, np.pi)
        if abs(sg.wrap(a - yaw0)) < np.deg2rad(55):
            continue  # keep the line of sight to W clear
        pillars.append([r * np.cos(a), r * np.sin(a), *rng.uniform(0.2, 0.45, 2), rng.uniform(-np.pi, np.pi)])
    return yaw0, wall, side * theta, np.array(pillars)


def wall_visible(pos, yaw, wall, others):
    """Is any point of W's front face inside the field of view and unoccluded?"""
    c, s = np.cos(wall[4]), np.sin(wall[4])
    pts = wall[:2] + np.linspace(-wall[2], wall[2], 21)[:, None] * np.array([c, s])
    boxes = np.concatenate([others, wall[None]])
    rr = sg.raycast2d(pos[:2], np.stack([np.cos(yaw + sg.RAY_ANG), np.sin(yaw + sg.RAY_ANG)], -1), boxes, sg.D_MAX + 1)
    return bool(sg.visibility(pts, pos, yaw, rr).any())


def rollout(yaw0, wall, turn, pillars, gap, variant):
    T = FACE + TURN + gap + TURN + HOVER
    phase = np.repeat([0, 1, 2, 3, 4], [FACE, TURN, gap, TURN, HOVER])
    drone = KinDrone((0.0, 0.0), yaw0)
    goal = 10.0 * np.array([np.cos(yaw0), np.sin(yaw0)])  # behind W, as in a normal flight (enters the state only)
    out = {k: [] for k in ["depth", "state", "action", "pos", "yaw", "wall_px"]}
    with_w = np.concatenate([pillars, wall[None]])
    for t in range(T):
        present = {"seen": True, "absent": False, "novel": phase[t] >= 2, "removed": phase[t] < 2}[variant]
        boxes = with_w if present else pillars
        yaw_cmd = yaw0 + (turn if 1 <= phase[t] <= 2 else 0.0)
        act = np.array([0, 0, 0, np.cos(yaw_cmd), np.sin(yaw_cmd)], np.float32)
        depth = drone.depth(boxes)
        out["wall_px"].append(int((np.abs(drone.depth(with_w) - drone.depth(pillars)) > 0.05).sum()))
        out["depth"].append(depth.astype(np.float16)); out["state"].append(drone.state(goal))
        out["action"].append(act); out["pos"].append(drone.pos.astype(np.float32)); out["yaw"].append(np.float32(drone.yaw))
        drone.step(act)
    return {k: np.stack(v) for k, v in out.items()} | {"phase": phase.astype(np.int8)}


def job(i):
    rng = np.random.default_rng([args.seed, i])
    while True:  # W must be out of view during the whole gap (checked at the end of the turn-away)
        yaw0, wall, turn, pillars = scenario(rng)
        d = KinDrone((0.0, 0.0), yaw0)
        for _ in range(TURN):
            d.step([0, 0, 0, np.cos(yaw0 + turn), np.sin(yaw0 + turn)])
        if not wall_visible(d.pos, d.yaw, wall, pillars):
            break
    for gap in GAPS:
        for variant in ("seen", "novel", "removed", "absent"):
            r = rollout(yaw0, wall, turn, pillars, gap, variant)
            np.savez_compressed(os.path.join(args.out, f"sc_{i:03d}_gap{gap:03d}_{variant}.npz"), **r,
                                wall=wall.astype(np.float32), pillars=pillars.astype(np.float32), gap=gap)
    return i


if __name__ == "__main__":
    os.makedirs(args.out, exist_ok=True)
    with mp.Pool(args.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(job, range(args.scenarios)), 1):
            if n % 10 == 0:
                print(f"scenarios {n}/{args.scenarios}", flush=True)
    print("MEMTEST_DATA_DONE", flush=True)
