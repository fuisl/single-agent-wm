"""Model-side helpers for scripts/viz (read-only use of lewm/): load a LeWM / R-LeWM checkpoint, embed a twin
memory-test rollout, compute its one-step latent prediction errors and run the belief (and M4's latent map).

The error logic is a copy of lewm/memtest.py's embed() / errors() (that script parses arguments at import):
frames every frameskip steps from step 0; window models predict frame j from the history_size frames before it,
belief models from b_{j-1} with the one-step open-loop head; e_j = squared error, normalised by target_var.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "lewm"))
from jepa import BeliefJEPA, TokenJEPA, encode_tokens  # noqa: E402
from utils import D_MAX, D_MIN, ZScore, action_blocks, depth_to_pixels, load_ckpt  # noqa: E402


class Model:
    def __init__(self, run_dir, device="cpu"):
        ck = load_ckpt(os.path.join(run_dir, "lewm_object.ckpt"), device)
        self.m, cfg, stats = ck["model"], ck["cfg"], ck["stats"]
        self.kind = "belief" if isinstance(self.m, BeliefJEPA) else "token" if isinstance(self.m, TokenJEPA) else "lewm"
        self.k, self.HS, self.dev = cfg["data"]["frameskip"], cfg["history_size"], device
        self.snorm = ZScore(*[s.to(device) for s in stats["state"]])
        self.anorm = ZScore(*[s.to(device) for s in stats["action"]])
        mt = os.path.join(run_dir, "memtest.json")
        self.target_var = json.load(open(mt))["target_var"] if os.path.exists(mt) else None
        self.is_map = self.kind == "belief" and hasattr(self.m.belief, "warp")

    def frames(self, d):
        """Frame-level arrays of a memtest rollout / episode dict (depth in metres)."""
        depth = np.round((np.asarray(d["depth"], np.float32) - D_MIN) / (D_MAX - D_MIN) * 255).astype(np.uint8)
        fr = np.arange(0, len(depth), self.k)
        return fr, depth[fr], np.asarray(d["state"])[fr], action_blocks(np.asarray(d["action"]), fr, self.k)

    @torch.no_grad()
    def embed(self, depth, state, blocks):
        px = depth_to_pixels(depth)[None].to(self.dev)
        s = self.snorm(torch.as_tensor(state, device=self.dev))
        a = self.anorm(torch.as_tensor(blocks, device=self.dev))
        if self.kind == "lewm":
            z = self.m.encode({"pixels": px, "state": s[None]})["emb"][0][:, None]
        else:
            z = encode_tokens(self.m.encoder, self.m.projector, px, self.m.spatial)[0]
        return z, s, a

    @torch.no_grad()
    def observe(self, z, s, a):
        """Beliefs (F, Bd) and, for M4, latent maps (F, C, S, S); None for window models."""
        if self.kind != "belief":
            return None, None
        a_prev = torch.cat([torch.zeros_like(a[:1]), a[:-1]])
        if self.is_map:
            b, m = self.m.belief.observe(z[None], s[None], a_prev[None], return_map=True)
            return b[0], m[0]
        return self.m.belief.observe(z[None], s[None], a_prev[None])[0], None

    @torch.no_grad()
    def errors(self, z, s, a):
        """(F,) one-step latent prediction error per frame (NaN where undefined), unnormalised."""
        F = len(z)
        e = torch.full((F,), float("nan"), device=self.dev)
        if self.kind == "belief":
            b, _ = self.observe(z, s, a)
            tok, _, _ = self.m.belief.imagine(b[:-1], a[:-1, None])
            e[1:] = (tok[:, 0] - z[1:]).pow(2).mean((-1, -2))
            return e.cpu().numpy()
        for j in range(self.HS, F):
            ctx = slice(j - self.HS, j)
            if self.kind == "lewm":
                pr = self.m.predict(z[None, ctx, 0], self.m.action_encoder(a[None, ctx]))[0, -1]
                e[j] = (pr - z[j, 0]).pow(2).mean()
            else:
                pr, _ = self.m.predict(z[None, ctx], s[None, ctx], self.m.action_encoder(a[None, ctx]))
                e[j] = (pr[0, -1] - z[j]).pow(2).mean()
        return e.cpu().numpy()
