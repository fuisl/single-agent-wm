"""collect.py without Isaac: same scripted pilot, labels and episode files, kinematic drone (kinsim.py).

    python kin_collect.py --workers 12 --episodes 1000 --out data/kin_v1
Episodes are independent (rng seeded by (seed, episode index)) and run in a process pool.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import scenegen as sg
from kinsim import KinDrone

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=12)
parser.add_argument("--episodes", type=int, default=2000)
parser.add_argument("--out", default="data/kin_v1")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_steps", type=int, default=1200)          # 60 s at 20 Hz
parser.add_argument("--glance_rate", type=float, default=0.0, help="scenegen.Behaviour look-away glances (kin_v2)")
args = parser.parse_args()


def episode(idx):
    rng = np.random.default_rng([args.seed, idx])
    mode = rng.choice(["goal", "explore", "trap"], p=[0.7, 0.15, 0.15])
    scene = sg.sample_scene(rng, need_trap=(mode == "trap"))
    pilot = sg.Behaviour(scene, mode, rng, glance_rate=args.glance_rate)
    drone = KinDrone(scene.start, np.arctan2(*(scene.goal - scene.start)[::-1]) + rng.uniform(-0.6, 0.6))
    seen = sg.SeenMap()
    d_prev = float(np.linalg.norm(scene.goal - scene.start))
    b = {k: [] for k in ["depth", "state", "action", "reward", "is_first", "is_last", "is_terminal",
                         "occ", "seen", "vis", "pos", "yaw"]}
    for t in range(args.max_steps):
        pos, yaw = drone.pos.copy(), drone.yaw
        _, occ, seen_m, vis = sg.labels(pos, yaw, scene.boxes, seen)
        r, d_prev, status, _ = sg.reward_terms(pos, scene.goal, d_prev, scene.boxes)
        if t == 0:
            r, status = 0.0, None
        out_of_arena = np.abs(pos[:2]).max() > sg.EXTENT
        last = status is not None or t + 1 >= args.max_steps or out_of_arena
        act = pilot.act(pos, yaw, t * 0.05) if not last else np.zeros(5, np.float32)
        b["depth"].append(drone.depth(scene.boxes).astype(np.float16)); b["state"].append(drone.state(scene.goal))
        b["action"].append(np.asarray(act, np.float32)); b["reward"].append(np.float32(r))
        b["is_first"].append(t == 0); b["is_last"].append(last); b["is_terminal"].append(status is not None)
        b["occ"].append(occ); b["seen"].append(seen_m); b["vis"].append(vis)
        b["pos"].append(pos.astype(np.float32)); b["yaw"].append(np.float32(yaw))
        if last:
            break
        drone.step(act)
    outcome = status or ("timeout" if t + 1 >= args.max_steps else "out_of_arena")
    fname = os.path.join(args.out, f"ep_{idx:05d}_kin_{mode}_{outcome}.npz")
    np.savez_compressed(fname, **{k: np.stack(v) for k, v in b.items()}, boxes=scene.boxes.astype(np.float32),
                        start=scene.start, goal=scene.goal, mode=mode, outcome=outcome)
    return str(mode), outcome, t + 1


if __name__ == "__main__":
    os.makedirs(args.out, exist_ok=True)
    outcomes, steps, t0 = {}, 0, time.time()
    with mp.Pool(args.workers) as pool:
        for n, (mode, outcome, length) in enumerate(pool.imap_unordered(episode, range(args.episodes)), 1):
            outcomes.setdefault(mode, {}).setdefault(outcome, 0)
            outcomes[mode][outcome] += 1
            steps += length
            if n % 20 == 0 or n == args.episodes:
                el = time.time() - t0
                info = {"episodes": n, "steps": steps, "steps_per_s": round(steps / el, 1),
                        "elapsed_min": round(el / 60, 1), "outcomes": outcomes}
                json.dump(info, open(os.path.join(args.out, "progress.json"), "w"), indent=1)
                print(json.dumps(info), flush=True)
    print("COLLECT_DONE", flush=True)
