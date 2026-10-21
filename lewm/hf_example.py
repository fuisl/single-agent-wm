"""Losses of a downloaded checkpoint on the example episodes, with the same forward as training (from lewm/):
    hf download fuisloy/r-lewm-uav --include "m1/*" "examples/*" --local-dir ../hf
    python hf_example.py --model ../hf/m1 --data ../hf/examples
"""
import argparse

import torch
from omegaconf import OmegaConf

from train import FORWARD  # registers the "eval" config resolver
from export_hf import load_pretrained
from module import SIGReg
from utils import WindowDataset, episode_files, load_episode


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats, cfg = load_pretrained(args.model, device)
    cfg = OmegaConf.create(cfg)
    print(f"{type(model).__name__}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    eps = [load_episode(f) for f in episode_files(args.data, "all")]
    data = WindowDataset(eps, cfg.data.num_steps, cfg.data.frameskip, stats)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True,
                                         generator=torch.Generator().manual_seed(0))
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device)
    forward = FORWARD[cfg.model._target_]

    sums, n = {}, 0
    with torch.no_grad():
        for _, batch in zip(range(args.batches), loader):
            out = forward(model, sigreg, {k: v.to(device) for k, v in batch.items()}, cfg)
            for k, v in out.items():
                if k.endswith("_loss") or k.startswith("belief_"):
                    sums[k] = sums.get(k, 0.0) + v.item()
            n += 1
    print(f"{len(eps)} episodes, {n} batches of {args.batch_size} windows ({cfg.data.num_steps} frames each):")
    for k, v in sums.items():
        print(f"  {k:16s} {v / n:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--batches", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    run(p.parse_args())
