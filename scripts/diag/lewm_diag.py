"""LeWM diagnostics on a trained checkpoint (run from lewm/ with PYTHONPATH=$PWD):
    python ../scripts/diag/lewm_diag.py --run_dir $WM_ROOT/runs/kin_v1/lewm --data $WM_ROOT/data/kin_v1
  A. where is the depth geometry? the same-budget depth decoder trained on train episodes, scored on held-out
     layouts, from: final emb / CLS before the state is added / state embedding alone / the 60 patch tokens,
     for the trained and a random-init encoder.
  B. what does the predictor use? next-embedding loss on val windows with the pixel or state part of the
     context replaced by its dataset mean, or the actions shuffled.
"""
import argparse
import json
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from module import DepthDecoder
from utils import ZScore, action_blocks, depth_to_pixels, episode_files, load_ckpt, load_episode, D_MIN, D_MAX

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--data", required=True)
p.add_argument("--n_train", type=int, default=120)
p.add_argument("--n_val", type=int, default=60)
p.add_argument("--stride", type=int, default=2)
p.add_argument("--steps", type=int, default=3000)
args = p.parse_args()
torch.manual_seed(0)
ckpt = load_ckpt(Path(args.run_dir) / "lewm_object.ckpt")
cfg, stats = ckpt["cfg"], ckpt["stats"]
rand = hydra.utils.instantiate(OmegaConf.create(cfg).model).cuda().eval().requires_grad_(False)
models = {"trained": ckpt["model"], "random": rand}
snorm = ZScore(*[s.cuda() for s in stats["state"]])
anorm = ZScore(*[s.cuda() for s in stats["action"]])
k, HS = cfg["data"]["frameskip"], cfg["history_size"]
rng = np.random.default_rng(0)
files = {"train": list(rng.permutation(episode_files(args.data, "train"))[:args.n_train]),
         "val": episode_files(args.data, "val")[:args.n_val]}
eps = {s: [load_episode(f) for f in fs] for s, fs in files.items()}


@torch.no_grad()
def parts(model, e, idx):
    hs, se = [], []
    for c in range(0, len(idx), 512):
        i = idx[c:c + 512]
        hs.append(model.encoder(depth_to_pixels(e["depth"][i]).cuda()).last_hidden_state)
        se.append(model.state_encoder(snorm(torch.as_tensor(e["state"][i], device="cuda"))[None])[0])
    hs, se = torch.cat(hs), torch.cat(se)
    cls = hs[:, 0]
    return {"emb": model.projector(cls + se), "cls": cls, "state": se,
            "patch": hs[:, 1:].reshape(len(idx), 6, 10, -1).permute(0, 3, 1, 2)}


class PatchDecoder(nn.Module):
    """(192, 6, 10) patch grid -> 48x80 depth, the same up-sampling stack as DepthDecoder."""

    def __init__(self, dim, ch=64):
        super().__init__()
        up = lambda i, o: [nn.ConvTranspose2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.GELU()]
        self.net = nn.Sequential(nn.Conv2d(dim, 4 * ch, 1), *up(4 * ch, 2 * ch), *up(2 * ch, ch), *up(ch, ch),
                                 nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)[:, 0])


# ---------------------------------------------------------------- A. depth decoding train -> held-out layouts
results = {"decode": {}, "predict": {}}
for mname, model in models.items():
    feats = {s: {} for s in eps}
    target = {}
    for s, es in eps.items():
        out, tgt = {}, []
        for e in es:
            idx = np.arange(0, len(e["reward"]), args.stride)
            for kk, v in parts(model, e, idx).items():
                out.setdefault(kk, []).append(v.half().cpu())
            tgt.append(torch.from_numpy(e["depth"][idx]))
        feats[s] = {kk: torch.cat(v) for kk, v in out.items()}
        target[s] = torch.cat(tgt)
    for rep in (["emb", "cls", "state", "patch"] if mname == "trained" else ["emb", "patch"]):
        dec = (PatchDecoder(192) if rep == "patch" else DepthDecoder(192)).cuda()
        opt = torch.optim.AdamW(dec.parameters(), lr=3e-4, weight_decay=1e-4)
        X, Y = feats["train"][rep], target["train"]
        for step in range(args.steps):
            i = torch.randint(len(X), (256,))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rec = dec(X[i].cuda().float()).float()
            loss = (rec - Y[i].cuda().float() / 255).abs().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        dec.eval()
        row = {}
        for s in ("train", "val"):
            err, near = [], []
            with torch.no_grad():
                for c in range(0, len(feats[s][rep]), 256):
                    rec = dec(feats[s][rep][c:c + 256].cuda().float()).float()
                    true = target[s][c:c + 256].cuda().float() / 255
                    a = (rec - true).abs() * (D_MAX - D_MIN)
                    err.append(a.mean((1, 2)).cpu())
                    m = true * (D_MAX - D_MIN) + D_MIN < 2.5  # pixels on obstacles closer than 2.5 m
                    near.append(torch.stack([(a * m).sum(), m.sum()]).cpu())
            near = torch.stack(near).sum(0)
            row[s] = {"mae_m": round(torch.cat(err).mean().item(), 3), "near_mae_m": round((near[0] / near[1]).item(), 3)}
        results["decode"][f"{mname}/{rep}"] = row
        print(json.dumps({f"{mname}/{rep}": row}), flush=True)
    del feats

# ---------------------------------------------------------------- B. what the trained predictor relies on
model = models["trained"]
W = []
for e in eps["val"]:
    T = len(e["reward"])
    for s in range(0, T - HS * k - k + 1, 7):
        W.append((e, s + np.arange(HS + 1) * k))
mean_cls, mean_se = None, None
with torch.no_grad():
    allp = [parts(model, e, t) for e, t in W]
    cls = torch.stack([q["cls"] for q in allp]); se = torch.stack([q["state"] for q in allp])  # (N, HS+1, D)
    act = torch.stack([anorm(torch.as_tensor(action_blocks(e["action"], t, k), device="cuda")) for e, t in W])
    mean_cls, mean_se = cls.flatten(0, 1).mean(0), se.flatten(0, 1).mean(0)
    proj = lambda c, s_: model.projector((c + s_).flatten(0, 1)).view(c.shape)
    emb = proj(cls, se)
    tgt = emb[:, HS]
    var = (tgt - tgt.mean(0)).pow(2).mean().item()

    def pred_loss(ctx, a):
        out = []
        for c in range(0, len(ctx), 1024):
            pr = model.predict(ctx[c:c + 1024], model.action_encoder(a[c:c + 1024]))[:, -1]
            out.append((pr - tgt[c:c + 1024]).pow(2).mean(1))
        return torch.cat(out).mean().item()

    ctx_a = act[:, :HS]
    rows = {
        "full context": pred_loss(emb[:, :HS], ctx_a),
        "pixels -> mean (state kept)": pred_loss(proj(mean_cls.expand_as(cls[:, :HS]), se[:, :HS]), ctx_a),
        "state -> mean (pixels kept)": pred_loss(proj(cls[:, :HS], mean_se.expand_as(se[:, :HS])), ctx_a),
        "actions shuffled": pred_loss(emb[:, :HS], ctx_a[torch.randperm(len(ctx_a))]),
        "copy last frame": (emb[:, HS - 1] - tgt).pow(2).mean().item(),
        "target variance": var,
    }
    # how much of the target itself is pixel- vs state-driven
    rows["target change if its pixels -> mean"] = (proj(mean_cls.expand_as(cls), se)[:, HS] - tgt).pow(2).mean().item()
    rows["target change if its state -> mean"] = (proj(cls, mean_se.expand_as(se))[:, HS] - tgt).pow(2).mean().item()
results["predict"] = {kk: round(v, 4) for kk, v in rows.items()}
results["predict"]["n_windows"] = len(W)
print(json.dumps(results["predict"], indent=1), flush=True)
json.dump(results, open(Path(args.run_dir) / "diag.json", "w"), indent=1)
print("DIAG_DONE", flush=True)
