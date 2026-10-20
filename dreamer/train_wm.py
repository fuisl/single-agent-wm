"""Offline world-model training on collected episodes with the official DreamerV3 (JAX) code,
plus feature extraction for the memory probe.

Runs in the dreamer env (.venvs/dreamer):
    python train_wm.py train   --data data/wm_v1 --logdir runs/wm50m --size size50m --steps 20000
    python train_wm.py extract --data data/wm_v1 --logdir runs/wm50m --size size50m

`train` samples random windows of batch_length steps (RSSM state reset at every window start) and runs
the unmodified DreamerV3 loss; the actor-critic part trains on imagination too but is not used.
`extract` replays whole episodes through the encoder + RSSM (posterior, carry kept across chunks) and
saves per-step features, plus an open-loop test: observe the first half of each chunk, imagine the
second half with the recorded actions, decode depth.
"""
import argparse
import glob
import json
import os
import pathlib
import pickle
import sys
import time
import zlib

DV3 = pathlib.Path(os.environ["DV3_ROOT"])       # set by env.sh
sys.path.insert(0, str(DV3))

import elements
import embodied
import jax
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml
from dreamerv3.agent import Agent

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["train", "extract"])
p.add_argument("--data", required=True)
p.add_argument("--logdir", required=True)
p.add_argument("--size", default="size50m")
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--batch_length", type=int, default=64)
p.add_argument("--chunk", type=int, default=64, help="extract: steps per report call")
p.add_argument("--val_frac", type=float, default=0.1)
p.add_argument("--max_extract_train", type=int, default=600, help="train episodes to extract for probe fitting")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--random_init", action="store_true", help="extract: skip loading weights (untrained baseline)")
p.add_argument("--feat_dir", default="features")
p.add_argument("--map_scale", type=float, default=0.0,
               help="weight of the auxiliary occupancy loss on seen cells (0 = plain DreamerV3)")
p.add_argument("--map_pos_weight", type=float, default=5.0)
args = p.parse_args()
logdir = pathlib.Path(args.logdir)
logdir.mkdir(parents=True, exist_ok=True)

D_MIN, D_MAX = 0.6, 6.0


# ---------------------------------------------------------------- data
def is_val(path):
    return (zlib.crc32(os.path.basename(path).encode()) % 1000) < args.val_frac * 1000


def load_episode(path):
    d = np.load(path)
    depth = np.round((d["depth"].astype(np.float32) - D_MIN) / (D_MAX - D_MIN) * 255).astype(np.uint8)[..., None]
    return {
        "depth": depth,
        "state": d["state"].astype(np.float32),
        "reward": d["reward"].astype(np.float32),
        "is_first": d["is_first"].astype(bool),
        "is_last": d["is_last"].astype(bool),
        "is_terminal": d["is_terminal"].astype(bool),
        "action": d["action"].astype(np.float32),
        **({"occ": d["occ"].astype(bool), "seen": d["seen"].astype(bool)} if args.map_scale > 0 else {}),
    }


files = sorted(glob.glob(os.path.join(args.data, "ep_*.npz")))
train_files = [f for f in files if not is_val(f)]
val_files = [f for f in files if is_val(f)]
print(f"{len(files)} episodes: {len(train_files)} train / {len(val_files)} val", flush=True)

obs_space = {
    "depth": elements.Space(np.uint8, (48, 80, 1)),
    "state": elements.Space(np.float32, (15,)),
    "reward": elements.Space(np.float32),
    "is_first": elements.Space(bool),
    "is_last": elements.Space(bool),
    "is_terminal": elements.Space(bool),
}
act_space = {"action": elements.Space(np.float32, (5,), -1.0, 1.0)}
MAP_KEYS = ("occ", "seen")          # privileged labels: used by the map loss only, never encoded or decoded
if args.map_scale > 0:
    obs_space.update({k: elements.Space(bool, (64, 64)) for k in MAP_KEYS})


# ---------------------------------------------------------------- agent
class ProbeAgent(Agent):
    """DreamerV3 agent whose `report` returns RSSM features and open-loop predictions instead of videos."""

    def report(self, carry, data):
        carry, obs, prevact, _ = self._apply_replay_context(carry, data)
        enc_carry, dyn_carry, dec_carry = carry
        reset = obs["is_first"]
        B, T = reset.shape
        kw = dict(training=False)
        enc_carry, _, tokens = self.enc(enc_carry, obs, reset, **kw)
        new_dyn, _, feat = self.dyn.observe(dyn_carry, tokens, prevact, reset, **kw)
        _, _, recons = self.dec(dec_carry, feat, reset, **kw)
        # open loop: posterior on the first half, prior-only imagination on the second half
        H = T // 2
        first = lambda xs: jax.tree.map(lambda x: x[:, :H], xs)
        second = lambda xs: jax.tree.map(lambda x: x[:, H:], xs)
        half_carry, _, _ = self.dyn.observe(dyn_carry, first(tokens), first(prevact), first(reset), **kw)
        _, imgfeat, _ = self.dyn.imagine(half_carry, second(prevact), length=T - H, training=False)
        _, _, imgrecons = self.dec(dec_carry, imgfeat, jnp.zeros_like(second(reset)), **kw)
        out = {
            "deter": feat["deter"].astype(jnp.float16),
            "stoch": feat["stoch"].reshape((B, T, -1)).astype(jnp.float16),
            "recon_depth": recons["depth"].pred()[..., 0].astype(jnp.float16),
            "img_depth": imgrecons["depth"].pred()[..., 0].astype(jnp.float16),
        }
        carry = (enc_carry, new_dyn, dec_carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, out


import ninjax as nj
import optax
from embodied.jax import nets


class MapHead(nj.Module):
    units: int = 1024

    def __init__(self):
        pass

    def __call__(self, x):
        x = self.sub("mlp", nets.MLP, 2, self.units)(x)
        return self.sub("out", nets.Linear, 64 * 64)(x)


class MapAgent(ProbeAgent):
    """DreamerV3 + auxiliary head that decodes the ego-centric occupancy map from (h_t, z_t).

    The loss is a positive-weighted BCE averaged over cells that have been seen so far (visible now or
    remembered); never-seen cells are not supervised. Gradients flow into the RSSM.
    """

    def __init__(self, obs_space, act_space, config):
        base = {k: v for k, v in obs_space.items() if k not in MAP_KEYS}
        orig = embodied.jax.Optimizer
        holder = {}

        def make_opt(modules, opt, **kw):                 # add the map head to the world-model optimizer
            holder["head"] = MapHead(name="maphead")
            return orig(list(modules) + [holder["head"]], opt, **kw)

        embodied.jax.Optimizer = make_opt
        try:
            super().__init__(base, act_space, config)
        finally:
            embodied.jax.Optimizer = orig
        self.maphead = holder["head"]
        self.obs_space = obs_space

    def loss(self, carry, obs, prevact, training):
        loss, (carry, entries, outs, metrics) = super().loss(carry, obs, prevact, training)
        B, T = obs["is_first"].shape
        logits = self.maphead(self.feat2tensor(outs["repfeat"])).astype(jnp.float32).reshape(B, T, -1)
        tgt = obs["occ"].reshape(B, T, -1).astype(jnp.float32)
        mask = obs["seen"].reshape(B, T, -1).astype(jnp.float32)
        bce = optax.sigmoid_binary_cross_entropy(logits, tgt) * (1 + (args.map_pos_weight - 1) * tgt)
        lmap = (bce * mask).sum(-1) / jnp.maximum(mask.sum(-1), 1.0)
        metrics["loss/map"] = lmap.mean()
        return loss + args.map_scale * lmap.mean(), (carry, entries, outs, metrics)


def make_agent(report_length):
    configs = yaml.YAML(typ="safe").load((DV3 / "dreamerv3" / "configs.yaml").read_text())
    config = elements.Config(configs["defaults"]).update(configs[args.size])
    config = config.update({
        "jax.prealloc": False, "jax.enable_policy": False, "replay_context": 0,
        "batch_size": args.batch_size, "batch_length": args.batch_length, "report_length": report_length,
        "seed": args.seed, "logdir": str(logdir),
    })
    cls = MapAgent if args.map_scale > 0 else ProbeAgent
    agent = cls(obs_space, act_space, elements.Config(
        **config.agent, logdir=config.logdir, seed=config.seed, jax=config.jax,
        batch_size=config.batch_size, batch_length=config.batch_length, replay_context=config.replay_context,
        report_length=config.report_length, replica=0, replicas=1))
    return agent, config


def ext_keys(batch, B, T):
    batch["consec"] = np.zeros((B, T), np.int32)
    batch["stepid"] = np.zeros((B, T, 20), np.uint8)
    return batch


def add_seed(agent, batch, counter):
    from embodied.jax import internal
    batch = internal.device_put(batch, agent.train_sharded)
    batch["seed"] = agent._seeds(counter, agent.train_mirrored)
    return batch


# ---------------------------------------------------------------- train
def train():
    agent, config = make_agent(report_length=args.chunk)
    rng = np.random.default_rng(args.seed)
    B, T = args.batch_size, args.batch_length
    print("loading episodes ...", flush=True)
    eps = [load_episode(f) for f in train_files]
    eps = [e for e in eps if len(e["reward"]) >= T]
    lengths = np.array([len(e["reward"]) for e in eps])
    probs = (lengths - T + 1) / (lengths - T + 1).sum()
    print(f"{len(eps)} train episodes, {lengths.sum()} steps", flush=True)
    log = open(logdir / "train_metrics.jsonl", "a")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.choice(len(eps), B, p=probs)
        batch = {k: [] for k in eps[0]}
        for i in idx:
            s = rng.integers(0, lengths[i] - T + 1)
            for k in batch:
                batch[k].append(eps[i][k][s:s + T])
        batch = {k: np.stack(v) for k, v in batch.items()}
        batch["is_first"][:, 0] = True
        batch = add_seed(agent, ext_keys(batch, B, T), step)
        carry = agent.init_train(B)
        _, _, mets = agent.train(carry, batch)
        if step % 200 == 0 and mets:
            keep = {k: float(np.asarray(v)) for k, v in mets.items()
                    if np.asarray(v).size == 1 and ("loss/" in k or k.endswith("/rec") or "ent" in k)}
            keep.update(step=step, minutes=round((time.time() - t0) / 60, 1))
            log.write(json.dumps(keep) + "\n"); log.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in keep.items()
                              if k in ("step", "minutes") or k.startswith("loss/")}), flush=True)
        if step % 5000 == 0 or step == args.steps:
            with open(logdir / "agent.pkl", "wb") as f:
                pickle.dump(agent.save(), f)
            print("saved checkpoint at step", step, flush=True)
    print("TRAIN_DONE", flush=True)


# ---------------------------------------------------------------- extract
def extract():
    B, T = args.batch_size, args.chunk
    agent, _ = make_agent(report_length=T)
    if not args.random_init:
        with open(logdir / "agent.pkl", "rb") as f:
            agent.load(pickle.load(f))
    outdir = logdir / args.feat_dir
    outdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    chosen = list(val_files) + list(rng.permutation(train_files)[:args.max_extract_train])
    counter = 0
    for g in range(0, len(chosen), B):
        group = chosen[g:g + B]
        eps = [load_episode(f) for f in group]
        while len(eps) < B:                       # pad the last group with copies
            eps.append(eps[0])
        L = max(len(e["reward"]) for e in eps)
        L = int(np.ceil(L / T) * T)
        padded = {}
        for k in eps[0]:
            arr = []
            for e in eps:
                x = e[k]
                pad = np.repeat(x[-1:], L - len(x), 0)
                arr.append(np.concatenate([x, pad], 0))
            padded[k] = np.stack(arr)
        carry = agent.init_report(B)
        outs = {k: [] for k in ("deter", "stoch", "recon_depth", "img_depth")}
        for s in range(0, L, T):
            batch = {k: v[:, s:s + T].copy() for k, v in padded.items()}
            batch = add_seed(agent, ext_keys(batch, B, T), counter); counter += 1
            carry, out = agent.report(carry, batch)
            for k in ("deter", "stoch", "recon_depth"):
                outs[k].append(np.asarray(out[k]))
            img = np.full((B, T, 48, 80), np.nan, np.float16)   # aligned: only the second half is imagined
            img[:, T // 2:] = np.asarray(out["img_depth"])
            outs["img_depth"].append(img)
        outs = {k: np.concatenate(v, 1) for k, v in outs.items()}
        for j, f in enumerate(group):
            n = len(eps[j]["reward"])
            name = os.path.basename(f).replace(".npz", "")
            np.savez(outdir / f"{name}.npz", deter=outs["deter"][j, :n], stoch=outs["stoch"][j, :n],
                     recon_depth=outs["recon_depth"][j, :n], img_depth=outs["img_depth"][j, :n],
                     split="val" if is_val(f) else "train", chunk=T)
        print(f"extracted {min(g + B, len(chosen))}/{len(chosen)}", flush=True)
    print("EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    train() if args.mode == "train" else extract()
