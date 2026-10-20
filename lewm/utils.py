import glob
import os
import zlib

import numpy as np
import torch

D_MIN, D_MAX = 0.6, 6.0


def is_val(path, frac=0.1):
    """Train/val split shared with probe.py and dreamer/train_wm.py: 10% of episodes by file-name hash."""
    return (zlib.crc32(os.path.basename(path).encode()) % 1000) < frac * 1000


def episode_files(data_dir, split):
    files = sorted(glob.glob(os.path.join(data_dir, "ep_*.npz")))
    assert files, f"no episodes in {data_dir}"
    return [f for f in files if is_val(f) == (split == "val")] if split != "all" else files


def load_episode(path):
    """depth as uint8 (same quantisation as the online env / DreamerV3), the rest as float32."""
    d = np.load(path)
    depth = np.round((d["depth"].astype(np.float32) - D_MIN) / (D_MAX - D_MIN) * 255).astype(np.uint8)
    return {
        "depth": depth,  # (T, 48, 80)
        "state": d["state"].astype(np.float32),  # (T, 15)
        "action": d["action"].astype(np.float32),  # (T, 5), a_t chosen after seeing obs_t
        "reward": d["reward"].astype(np.float32),
        "is_terminal": d["is_terminal"].astype(bool),
    }


def depth_to_pixels(depth):
    """uint8 depth (..., H, W) -> float pixels (..., 1, H, W) in [-1, 1]."""
    return (torch.as_tensor(depth).float() / 127.5 - 1.0).unsqueeze(-3)


def action_blocks(action, starts, frameskip):
    """Concatenate the frameskip actions following each frame start: (len(starts), frameskip * A)."""
    T = len(action)
    idx = np.asarray(starts)[:, None] + np.arange(frameskip)
    return action[np.minimum(idx, T - 1)].reshape(len(starts), -1)


class ZScore:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


def fit_stats(episodes, frameskip):
    state = np.concatenate([e["state"] for e in episodes])
    action = np.concatenate([action_blocks(e["action"], np.arange(len(e["action"])), frameskip) for e in episodes])
    stat = lambda x: (torch.tensor(x.mean(0)), torch.tensor(x.std(0) + 1e-6))
    return {"state": stat(state), "action": stat(action)}


class WindowDataset(torch.utils.data.Dataset):
    """Windows of num_steps frames, frameskip env steps apart, fully inside one episode (in RAM)."""

    def __init__(self, episodes, num_steps, frameskip, stats):
        self.episodes, self.num_steps, self.frameskip = episodes, num_steps, frameskip
        self.norm = {k: ZScore(*v) for k, v in stats.items()}
        span = (num_steps - 1) * frameskip + frameskip  # last frame needs its full action block
        self.index = [(i, s) for i, e in enumerate(episodes) for s in range(len(e["reward"]) - span + 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, s = self.index[k]
        e = self.episodes[i]
        t = s + np.arange(self.num_steps) * self.frameskip
        return {
            "pixels": depth_to_pixels(e["depth"][t]),
            "state": self.norm["state"](torch.from_numpy(e["state"][t])),
            "action": self.norm["action"](torch.from_numpy(action_blocks(e["action"], t, self.frameskip))),
        }


def save_ckpt(path, model, stats, cfg, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model, "stats": stats, "cfg": cfg, "step": step}, path)


def load_ckpt(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt["model"].eval().requires_grad_(False)
    return ckpt


@torch.no_grad()
def encode_steps(model, stats, depth, state, chunk=512):
    """Embeddings of every env step: depth (T, H, W) uint8, state (T, S) -> (T, D) on the model device."""
    device = next(model.parameters()).device
    norm = ZScore(*[s.to(device) for s in stats["state"]])
    out = []
    for k in range(0, len(depth), chunk):
        info = {"pixels": depth_to_pixels(depth[k:k + chunk])[None].to(device),
                "state": norm(torch.as_tensor(state[k:k + chunk], device=device))[None]}
        out.append(model.encode(info)["emb"][0])
    return torch.cat(out)
