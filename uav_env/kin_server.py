"""Drop-in replacement for rl_server.py with the kinematic drone (kinsim.py); no Isaac, numpy only.

Same protocol, arguments, scenes, observations, reward and termination as rl_server.py, so DreamerV3
(train_rl.py / eval_rl.py) and LeWM (lewm/eval.py) connect to it unchanged.
    python kin_server.py --num_envs 16 --port 6100
"""
import argparse
import time
from multiprocessing.connection import Listener, wait

import numpy as np

import scenegen as sg
from kinsim import KinDrone

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--port", type=int, default=6100)
parser.add_argument("--authkey", default="airdreamer")
parser.add_argument("--extras", action="store_true", help="also send pos / yaw (evaluation only, not in obs_space)")
parser.add_argument("--max_steps", type=int, default=1200)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--trap_start_frac", type=float, default=0.0,
                    help="training: fraction of random episodes that start inside a U or C trap")
args = parser.parse_args()
N = args.num_envs


class Slot:
    def __init__(self, i):
        self.i = i
        self.rng = np.random.default_rng(args.seed * 1000 + i)
        self.done = True

    def new_episode(self, options):
        rng = np.random.default_rng(options["seed"]) if options.get("seed") is not None else self.rng
        kind = options.get("kind", "random")
        if kind == "random" and rng.random() < args.trap_start_frac:
            self.scene = sg.sample_test_scene(rng, rng.choice(["utrap_in", "ctrap"]), jitter=True)
        elif kind == "random":
            self.scene = sg.sample_scene(rng)
        else:
            self.scene = sg.sample_test_scene(rng, kind)
        self.d_prev = float(np.linalg.norm(self.scene.goal - self.scene.start))
        self.t, self.done = 0, False
        yaw0 = self.scene.start_yaw if getattr(self.scene, "start_yaw", None) is not None else \
            np.arctan2(*(self.scene.goal - self.scene.start)[::-1]) + rng.uniform(-0.6, 0.6)
        self.drone = KinDrone(self.scene.start, yaw0)
        return np.array([0, 0, 0, np.cos(yaw0), np.sin(yaw0)], np.float32)   # hover right after the reset

    def obs(self):
        d = self.drone
        pos = d.pos.copy()
        first = self.t == 0
        r, self.d_prev, status, _ = sg.reward_terms(pos, self.scene.goal, self.d_prev, self.scene.boxes)
        if first:
            r, status = 0.0, None
        timeout = self.t + 1 >= args.max_steps or np.abs(pos[:2]).max() > sg.EXTENT
        last = status is not None or timeout
        depth = d.depth(self.scene.boxes)
        o = {
            "depth": np.round((depth - sg.D_MIN) / (sg.D_MAX - sg.D_MIN) * 255).astype(np.uint8)[..., None],
            "state": d.state(self.scene.goal),
            "reward": np.float32(r), "is_first": first, "is_last": last, "is_terminal": status is not None,
            "log/goal": np.float32(status == "goal"), "log/collision": np.float32(status == "collision"),
            "log/height": np.float32(status == "height"), "log/timeout": np.float32(last and status is None),
        }
        if args.extras:
            o["pos"], o["yaw"] = pos.astype(np.float32), np.float32(d.yaw)
        self.t += 1
        self.done = last
        return o


slots = [Slot(i) for i in range(N)]
listener = Listener(("127.0.0.1", args.port), authkey=args.authkey.encode(), backlog=128)
print(f"SERVER_READY port={args.port} envs={N} (kinematic)", flush=True)
conns = {}
while len(conns) < N:                                   # every client introduces itself with its slot index
    c = listener.accept()
    msg = c.recv()
    assert msg[0] == "hello", msg
    conns[int(msg[1])] = c
    print("client", msg[1], "connected", flush=True)

pending = {}
steps, t0 = 0, time.time()
while True:
    for c in wait(list(conns.values())):
        i = next(k for k, v in conns.items() if v is c)
        try:
            pending[i] = c.recv()
        except EOFError:
            print("client", i, "disconnected, shutting down", flush=True)
            raise SystemExit
    if len(pending) < N:
        continue
    for i, msg in pending.items():
        s = slots[i]
        if msg[0] == "reset" or s.done:
            a = s.new_episode(msg[1] if msg[0] == "reset" else {})
        else:
            a = msg[1]
        s.drone.step(a)
    for i in range(N):
        conns[i].send(slots[i].obs())
    pending.clear()
    steps += 1
    if steps % 500 == 0:
        print(f"server steps={steps} ({steps * N / (time.time() - t0):.0f} agent-steps/s)", flush=True)
