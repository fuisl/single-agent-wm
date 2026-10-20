"""Evaluate a trained online policy on fixed test scenes (runs in the dreamer env).

Start the server first with --extras (adds pos / yaw for trajectory logging):
    python rl_server.py --num_envs 16 --port 6140 --extras
    python eval_rl.py --logdir runs/dreamer_uav --port 6140 --episodes 100
Test suites (seeded, identical for every model): random (training distribution), utrap, utrap_in, ctrap.
Writes eval/<suite>.json with outcome rates and per-episode records incl. trajectories.
"""
import argparse
import json
import os
import pathlib
import sys
import time
from multiprocessing.connection import Client

DV3 = pathlib.Path(os.environ["DV3_ROOT"])       # set by env.sh
sys.path.insert(0, str(DV3))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import elements
import numpy as np
import ruamel.yaml as yaml
from dreamerv3.agent import Agent

from isaac_uav_env import IsaacUAV

p = argparse.ArgumentParser()
p.add_argument("--logdir", required=True)
p.add_argument("--port", type=int, default=6140)
p.add_argument("--envs", type=int, default=16)
p.add_argument("--episodes", type=int, default=100, help="per suite")
p.add_argument("--suites", default="random,utrap,utrap_in,ctrap")
p.add_argument("--seed_base", type=int, default=100000)
args = p.parse_args()

logdir = pathlib.Path(args.logdir)
config = elements.Config(yaml.YAML(typ="safe").load((logdir / "config.yaml").read_text()))
spaces = IsaacUAV(0)
obs_space = {k: v for k, v in spaces.obs_space.items() if not k.startswith("log/")}
act_space = {k: v for k, v in spaces.act_space.items() if k != "reset"}
config = config.update({"jax.prealloc": False})
agent = Agent(obs_space, act_space, elements.Config(
    **config.agent, logdir=str(logdir), seed=config.seed, jax=config.jax, batch_size=config.batch_size,
    batch_length=config.batch_length, replay_context=config.replay_context, report_length=config.report_length,
    replica=0, replicas=1))
cp = elements.Checkpoint(elements.Path(logdir / "ckpt"))
cp.agent = agent
cp.load(keys=["agent"])
print("loaded", cp.latest(), flush=True)

conns = []
for i in range(args.envs):
    c = Client(("127.0.0.1", args.port), authkey=b"airdreamer"); c.send(("hello", i)); conns.append(c)
outdir = logdir / "eval"
outdir.mkdir(exist_ok=True)

for suite_i, suite in enumerate(args.suites.split(",")):
    queue = [{"kind": suite, "seed": args.seed_base + 1000 * suite_i + k} for k in range(args.episodes)]
    current = [None] * args.envs
    records, traj = [], [[] for _ in range(args.envs)]
    carry = agent.init_policy(args.envs)
    for i, c in enumerate(conns):                       # first reset of every slot
        current[i] = queue.pop(0) if queue else None
        c.send(("reset", current[i] or {"kind": "random"}))
    t0 = time.time()
    while any(x is not None for x in current):
        obs = [c.recv() for c in conns]
        for i, o in enumerate(obs):
            if current[i] is not None:
                traj[i].append([*o["pos"].tolist(), float(o["yaw"])])
        batch = {k: np.stack([o[k] for o in obs]) for k in obs_space}
        carry, acts, _ = agent.policy(carry, batch, mode="eval")
        for i, (c, o) in enumerate(zip(conns, obs)):
            if o["is_last"]:
                if current[i] is not None:
                    outcome = next((k for k in ("goal", "collision", "height") if o[f"log/{k}"] > 0), "timeout")
                    records.append({**current[i], "outcome": outcome, "steps": len(traj[i]), "traj": traj[i]})
                current[i] = queue.pop(0) if queue else None
                traj[i] = []
                c.send(("reset", current[i] or {"kind": "random"}))
            else:
                c.send(("step", np.clip(acts["action"][i], -1, 1).astype(np.float32)))
    for c in conns:                                     # drain the replies to the last round so the next
        c.recv()                                        # suite starts in lock-step with the server
    n = len(records)
    rates = {k: sum(r["outcome"] == k for r in records) / n for k in ("goal", "collision", "height", "timeout")}
    ok = [r["steps"] / 20 for r in records if r["outcome"] == "goal"]
    summary = {"suite": suite, "episodes": n, **rates, "mean_time_to_goal_s": float(np.mean(ok)) if ok else None,
               "minutes": round((time.time() - t0) / 60, 1)}
    print("EVAL", json.dumps(summary), flush=True)
    json.dump({"summary": summary, "episodes": sorted(records, key=lambda r: r["seed"])},
              open(outdir / f"{suite}.json", "w"))
for c in conns:
    c.close()
print("EVAL_DONE", flush=True)
