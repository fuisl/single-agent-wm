"""Kinematic dry run of the behaviour policy (no Isaac): first-order velocity tracking at 200 Hz.

Reports outcome rates per mode so the scripted pilot can be tuned before the expensive collection.
    python dry_run.py --episodes 60
"""
import argparse
import collections
import time

import numpy as np

import scenegen as sg

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=60)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--trap_only", action="store_true")
args = p.parse_args()
rng = np.random.default_rng(args.seed)
stats = collections.defaultdict(collections.Counter)
lengths = collections.defaultdict(list)
t0 = time.time()
for ep in range(args.episodes):
    mode = rng.choice(["goal", "explore", "trap"], p=[0.7, 0.15, 0.15]) if not args.trap_only else "trap"
    scene = sg.sample_scene(rng, need_trap=(mode == "trap"))
    pilot = sg.Behaviour(scene, mode, rng)
    pos = np.array([*scene.start, 1.5]); vel = np.zeros(3)
    yaw = np.arctan2(*(scene.goal - scene.start)[::-1]); d_prev = np.linalg.norm(scene.goal - scene.start)
    seen = sg.SeenMap()
    status = "timeout"
    for step in range(1200):
        _, occ, seen_m, vis = sg.labels(pos, yaw, scene.boxes, seen)
        r, d_prev, st, _ = sg.reward_terms(pos, scene.goal, d_prev, scene.boxes)
        if st:
            status = st; break
        a = pilot.act(pos, yaw, step * 0.05)
        yaw_cmd = np.arctan2(a[4], a[3])
        for _ in range(10):
            c, s = np.cos(yaw), np.sin(yaw); vb = a[:3] * sg.LAMBDA
            vw = np.array([c * vb[0] - s * vb[1], s * vb[0] + c * vb[1], vb[2]])
            vel += (vw - vel) * 0.005 / 0.25; pos = pos + vel * 0.005
            yaw = sg.wrap(yaw + np.clip(sg.wrap(yaw_cmd - yaw) / 0.3, -2.5, 2.5) * 0.005)
    if status == "collision": print("  collision", mode, "step", step, "pos", pos.round(2), "look", pilot.look_timer, "targets left", len(pilot.targets))
    stats[mode][status] += 1; lengths[mode].append(step)
print(f"{args.episodes} episodes in {time.time() - t0:.0f}s")
for m in stats:
    print(m, dict(stats[m]), "mean len", int(np.mean(lengths[m])))
