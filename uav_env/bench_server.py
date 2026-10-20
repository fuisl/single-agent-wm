"""Throughput check for rl_server.py with random actions (no learning). Runs in any env.

    python bench_server.py --num_envs 16 --port 6113 --rounds 1500
"""
import argparse
import time
from multiprocessing.connection import Client

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--port", type=int, default=6100)
p.add_argument("--rounds", type=int, default=1500)
args = p.parse_args()
conns = []
for i in range(args.num_envs):
    c = Client(("127.0.0.1", args.port), authkey=b"airdreamer"); c.send(("hello", i)); conns.append(c)
rng = np.random.default_rng(0)
done = [True] * args.num_envs
outcomes, t0 = {}, time.time()
for r in range(args.rounds):
    for i, c in enumerate(conns):
        if done[i]:
            c.send(("reset", {}))
        else:
            a = rng.uniform(-1, 1, 5).astype(np.float32); a[0] = abs(a[0]) * 0.6
            c.send(("step", a))
    for i, c in enumerate(conns):
        o = c.recv(); done[i] = bool(o["is_last"])
        if done[i]:
            k = [n for n in ("goal", "collision", "height", "timeout") if o[f"log/{n}"] > 0]
            outcomes[k[0] if k else "?"] = outcomes.get(k[0] if k else "?", 0) + 1
el = time.time() - t0
print(f"BENCH rounds={args.rounds} agent-steps/s={args.rounds * args.num_envs / el:.0f} outcomes={outcomes}")
