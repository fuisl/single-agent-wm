"""Offline LeWM training on collected UAV episodes (main env, run from lewm/):
    python train.py data.dir=$WM_ROOT/data/wm_v1 run_dir=$WM_ROOT/runs/lewm trainer.max_steps=30000
Same loss as the official train.py (next-embedding MSE + SIGReg); a plain torch loop replaces lightning/spt.
"""
import json
import math
import time
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import WindowDataset, episode_files, fit_stats, load_episode, save_ckpt

OmegaConf.register_new_resolver("eval", eval)


def lejepa_forward(model, sigreg, batch, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    output = model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
    return output


def warmup_cosine(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    torch.manual_seed(cfg.seed)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    #########################
    ##       dataset       ##
    #########################

    load = lambda split: [load_episode(f) for f in episode_files(cfg.data.dir, split)]
    train_eps, val_eps = load("train"), load("val")
    print(f"{len(train_eps)} train / {len(val_eps)} val episodes", flush=True)
    stats = fit_stats(train_eps, cfg.data.frameskip)
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = cfg.data.frameskip * train_eps[0]["action"].shape[-1]
        cfg.model.state_encoder.input_dim = train_eps[0]["state"].shape[-1]
    OmegaConf.save(cfg, run_dir / "config.yaml")

    make = lambda eps: WindowDataset(eps, cfg.data.num_steps, cfg.data.frameskip, stats)
    train_set, val_set = make(train_eps), make(val_eps)
    gen = torch.Generator().manual_seed(cfg.seed)
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=True, drop_last=False, generator=gen)

    ##############################
    ##       model / optim      ##
    ##############################

    model = hydra.utils.instantiate(cfg.model).cuda()
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).cuda()
    print(f"LeWM params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    total = cfg.trainer.max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, cfg.trainer.warmup_steps, total))
    to_cuda = lambda b: {k: v.cuda(non_blocking=True) for k, v in b.items()}
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.trainer.precision == "bf16")

    ##########################
    ##       training       ##
    ##########################

    log = open(run_dir / "train_metrics.jsonl", "a")
    step, t0 = 0, time.time()
    while step < total:
        for batch in train:
            model.train()
            with autocast():
                out = lejepa_forward(model, sigreg, to_cuda(batch), cfg)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clip_val)
            opt.step()
            sched.step()
            step += 1

            if step % cfg.trainer.log_every == 0:
                rec = {f"train/{k}": out[k].item() for k in ("loss", "pred_loss", "sigreg_loss")}
                if step % cfg.trainer.val_every == 0:
                    model.eval()
                    with torch.no_grad(), autocast():
                        vals = [lejepa_forward(model, sigreg, to_cuda(b), cfg) for _, b in zip(range(20), val)]
                    rec.update({f"val/{k}": sum(v[k].item() for v in vals) / len(vals) for k in ("pred_loss", "sigreg_loss")})
                rec.update(step=step, lr=sched.get_last_lr()[0], minutes=round((time.time() - t0) / 60, 1))
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps({k: round(v, 4) if isinstance(v, float) and k != "lr" else v for k, v in rec.items()}), flush=True)
            if step % cfg.trainer.save_every == 0 or step == total:
                save_ckpt(run_dir / f"{cfg.output_model_name}_object.ckpt", model, stats, OmegaConf.to_container(cfg), step)
            if step == total:
                break
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    run()
