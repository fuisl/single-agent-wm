"""LeWM + CEM model-predictive control in Isaac Sim, in place of DreamerV3's actor (main env, run from lewm/).

Start the server first with --extras (pos / yaw for trajectory logging):
    $PY_ISAAC $WM_ROOT/uav_env/rl_server.py --num_envs 16 --port 6140 --extras
    python eval.py run_dir=$WM_ROOT/runs/lewm port=6140 episodes=100
Same seeded test suites and output format as dreamer/eval_rl.py: <run_dir>/eval_cem/<suite>.json.

Every action_block rounds the current observations of all drones are encoded and appended to their embedding
histories; drones whose queued actions ran out are replanned together (one batched CEM): `horizon` blocks from
the last history_size embeddings, of which the first receding_horizon blocks are executed. After a reset a drone
hovers until the next planning round (< 0.25 s) so that all drones share the planning clock.
"""
import json
import time
from multiprocessing.connection import Client
from pathlib import Path

import hydra
import numpy as np
import torch

from utils import ZScore, depth_to_pixels, load_ckpt

HOVER = np.array([0, 0, 0, 1, 0], np.float32)


class Slot:
    def __init__(self):
        self.reset()

    def reset(self):
        self.emb, self.act, self.block, self.queue, self.mean = [], [], [], [], None


@hydra.main(version_base=None, config_path="./config/eval", config_name="uav")
def run(cfg):
    run_dir = Path(cfg.run_dir)
    ckpt = load_ckpt(run_dir / "lewm_object.ckpt")
    model, stats, tcfg = ckpt["model"], ckpt["stats"], ckpt["cfg"]
    heads = torch.load(run_dir / "heads.pt", weights_only=False)["head"]
    k, HS, P, R = tcfg["data"]["frameskip"], tcfg["history_size"], cfg.plan_config.horizon, cfg.plan_config.receding_horizon
    assert cfg.plan_config.action_block == k, "action_block must equal the training frameskip"
    solver = hydra.utils.instantiate(cfg.solver, model=model, heads=heads, history_size=HS)
    s_norm = ZScore(*[s.cuda() for s in stats["state"]])
    a_norm = ZScore(*[s.cuda() for s in stats["action"]])
    low, high = a_norm(-torch.ones(5 * k, device="cuda")), a_norm(torch.ones(5 * k, device="cuda"))

    def plan(slots, obs):
        """Encode the new frames of `slots`, then replan for those whose action queue is empty."""
        info = {"pixels": depth_to_pixels(np.stack([o["depth"][..., 0] for o in obs]))[:, None].cuda(),
                "state": s_norm(torch.as_tensor(np.stack([o["state"] for o in obs]), device="cuda"))[:, None]}
        emb = model.encode(info)["emb"][:, 0]
        for s, e in zip(slots, emb):
            s.emb = (s.emb + [e])[-HS:]
        need = [s for s in slots if not s.queue]
        for H in {len(s.emb) for s in need}:  # batch slots with the same history length
            group = [s for s in need if len(s.emb) == H]
            e = torch.stack([torch.stack(s.emb) for s in group])
            a = torch.stack([torch.stack(s.act[-(H - 1):]) if H > 1 else torch.zeros(0, 5 * k, device="cuda")
                             for s in group])
            init = None if any(s.mean is None for s in group) else torch.stack([s.mean for s in group])
            mean = solver.solve(e, a, P, low, high, init)
            for s, m in zip(group, mean):
                s.mean = torch.cat([m[R:], torch.zeros_like(m[:R])])  # warm start: shift by executed blocks
                s.queue = list(a_norm.inverse(m[:R]).clamp(-1, 1).view(R * k, 5).cpu().numpy())

    conns = []
    for i in range(cfg.envs):
        c = Client(("127.0.0.1", cfg.port), authkey=b"airdreamer"); c.send(("hello", i)); conns.append(c)
    outdir = run_dir / "eval_cem"
    outdir.mkdir(exist_ok=True)

    for suite_i, suite in enumerate(cfg.suites.split(",")):
        queue = [{"kind": suite, "seed": cfg.seed_base + 1000 * suite_i + n} for n in range(cfg.episodes)]
        current = [None] * cfg.envs
        slots = [Slot() for _ in range(cfg.envs)]
        records, traj = [], [[] for _ in range(cfg.envs)]
        for i, c in enumerate(conns):
            current[i] = queue.pop(0) if queue else None
            c.send(("reset", current[i] or {"kind": "random"}))
        t0, plan_time, rnd = time.time(), 0.0, 0
        while any(x is not None for x in current):
            obs = [c.recv() for c in conns]
            for i, o in enumerate(obs):
                if current[i] is not None:
                    traj[i].append([*o["pos"].tolist(), float(o["yaw"])])
            live = [i for i, o in enumerate(obs) if current[i] is not None and not o["is_last"]]
            if live and rnd % k == 0:  # every slot plans at the same rounds, so CEM runs batched over all drones
                tp = time.time()
                plan([slots[i] for i in live], [obs[i] for i in live])
                plan_time += time.time() - tp
            for i, (c, o) in enumerate(zip(conns, obs)):
                s = slots[i]
                if o["is_last"]:
                    if current[i] is not None:
                        outcome = next((n for n in ("goal", "collision", "height") if o[f"log/{n}"] > 0), "timeout")
                        records.append({**current[i], "outcome": outcome, "steps": len(traj[i]), "traj": traj[i]})
                    current[i] = queue.pop(0) if queue else None
                    traj[i] = []
                    s.reset()
                    c.send(("reset", current[i] or {"kind": "random"}))
                elif current[i] is None:
                    c.send(("step", HOVER))
                elif not s.emb:  # new episode: hover (holding yaw) until the next planning round, < 1 block
                    c.send(("step", np.array([0, 0, 0, np.cos(o["yaw"]), np.sin(o["yaw"])], np.float32)))
                else:
                    a = s.queue.pop(0)
                    s.block.append(a)
                    if len(s.block) == k:  # block finished: keep it (normalised) as predictor context
                        s.act = (s.act + [a_norm(torch.as_tensor(np.concatenate(s.block), device="cuda"))])[-HS:]
                        s.block = []
                    c.send(("step", a.astype(np.float32)))
            rnd += 1
        for c in conns:  # drain the replies to the last round so the next suite starts in lock-step
            c.recv()
        n = len(records)
        rates = {o: sum(r["outcome"] == o for r in records) / n for o in ("goal", "collision", "height", "timeout")}
        ok = [r["steps"] / 20 for r in records if r["outcome"] == "goal"]
        summary = {"suite": suite, "episodes": n, **rates, "mean_time_to_goal_s": float(np.mean(ok)) if ok else None,
                   "minutes": round((time.time() - t0) / 60, 1), "plan_minutes": round(plan_time / 60, 1)}
        print("EVAL", json.dumps(summary), flush=True)
        json.dump({"summary": summary, "episodes": sorted(records, key=lambda r: r["seed"])},
                  open(outdir / f"{suite}.json", "w"))
    for c in conns:
        c.close()
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    run()
