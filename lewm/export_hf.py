"""Hugging Face export in the official LeWM layout (weights.pt state dict + config.json), and its loader.
    python export_hf.py --ckpt $WM_ROOT/runs/kin_v1/m1/lewm_object.ckpt --out $WM_ROOT/hf/m1
    model, stats, cfg = load_pretrained("hf/m1")   # from lewm/, after `hf download`
config.json holds the resolved training config, the state / action z-score stats and the training step.
"""
import argparse
import json
import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


def export(ckpt_path, out):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.to_container(OmegaConf.create(ckpt["cfg"]), resolve=True)
    stats = {k: [t.tolist() for t in v] for k, v in ckpt["stats"].items()}
    torch.save(ckpt["model"].state_dict(), out / "weights.pt")
    text = json.dumps({"cfg": cfg, "stats": stats, "step": ckpt["step"]}, indent=1)
    if os.environ.get("WM_ROOT"):  # no local paths in the published config
        text = text.replace(os.environ["WM_ROOT"], "$WM_ROOT")
    (out / "config.json").write_text(text)

    model, _, _ = load_pretrained(out)  # round trip: same outputs as the pickled model
    ref = ckpt["model"].eval()
    x = {k: v.cpu() for k, v in example_batch(cfg, ckpt["stats"]).items()}
    with torch.no_grad():
        a, b = ref.encode(dict(x))["emb"], model.encode(dict(x))["emb"]
    assert torch.equal(a, b), (a - b).abs().max()
    print(f"{ckpt_path} -> {out} (step {ckpt['step']})")


def load_pretrained(path, device="cpu"):
    """(model in eval mode, stats {"state" | "action": (mean, std)}, resolved training cfg)."""
    meta = json.loads((Path(path) / "config.json").read_text())
    model = hydra.utils.instantiate(meta["cfg"]["model"])
    model.load_state_dict(torch.load(Path(path) / "weights.pt", map_location="cpu"), strict=True)
    stats = {k: tuple(torch.tensor(t) for t in v) for k, v in meta["stats"].items()}
    return model.to(device).eval().requires_grad_(False), stats, meta["cfg"]


def example_batch(cfg, stats, B=2):
    """Random normalised batch with the training shapes: pixels (B, T, 1, 48, 80), state (B, T, S), action (B, T, A)."""
    T, (H, W) = cfg["data"]["num_steps"], cfg["model"]["encoder"]["image_size"]
    return {"pixels": torch.rand(B, T, 1, H, W) * 2 - 1,
            "state": torch.randn(B, T, len(stats["state"][0])),
            "action": torch.randn(B, T, len(stats["action"][0]))}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    export(args.ckpt, args.out)
