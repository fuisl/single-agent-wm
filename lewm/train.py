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


def token_sigreg(sigreg, emb, cfg):
    """SIGReg per frame and token position ((T * N, B, D)) on a random subset of loss.sigreg.positions positions:
    an unbiased estimate of the mean over all positions at a fraction of the memory."""
    x = emb.permute(1, 2, 0, 3).flatten(0, 1)
    return sigreg(x[torch.randperm(len(x), device=x.device)[:cfg.loss.sigreg.positions]])


def token_forward(model, sigreg, batch, cfg):
    """M0 / M1: lejepa_forward over N tokens per frame, plus the next-state (proprio) prediction loss."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    output = model.encode(batch)
    emb = output["emb"]  # (B, T, N, D)
    state = batch["state"]

    pred_emb, pred_state = model.predict(emb[:, :ctx_len], state[:, :ctx_len], output["act_emb"][:, :ctx_len])
    output["pred_loss"] = (pred_emb - emb[:, n_preds:]).pow(2).mean()
    output["state_loss"] = (pred_state - state[:, n_preds:]).pow(2).mean()
    output["sigreg_loss"] = token_sigreg(sigreg, emb, cfg)
    output["loss"] = output["pred_loss"] + cfg.loss.state_weight * output["state_loss"] + lambd * output["sigreg_loss"]
    return output


def blind_mask(B, T, cfg, device):
    """M3: per window, with probability blind.prob, one span of blind.min..blind.max frames without the image."""
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for i in torch.nonzero(torch.rand(B) < cfg.blind.prob).flatten().tolist():
        n = int(torch.randint(cfg.blind.min, cfg.blind.max + 1, ()))
        s = int(torch.randint(1, max(T - n, 2), ()))
        mask[i, s:s + n] = True
    return mask


def belief_forward(model, sigreg, batch, cfg):
    """M2 / M3: closed-loop belief over the window, open-loop prediction of the next `horizon` frames from every
    position (BYOL-Explore style), next-token + next-state losses, SIGReg on the frame tokens."""

    lambd = cfg.loss.sigreg.weight
    output = model.encode(batch)
    emb, state, action = output["emb"], batch["state"], batch["action"]  # action[:, t]: block after frame t
    B, T = emb.shape[:2]
    K = model.horizon

    a_prev = torch.cat([torch.zeros_like(action[:, :1]), action[:, :-1]], 1)
    blind = blind_mask(B, T, cfg, emb.device) if cfg.blind.prob > 0 and model.training else None
    belief = model.belief.observe(emb, state, a_prev, blind)  # (B, T, Bd)

    idx = torch.arange(T, device=emb.device)[:, None] + torch.arange(K, device=emb.device)  # action index t + k - 1
    valid = (idx + 1 < T).float()  # target frame t + k inside the window
    tgt_i = (idx + 1).clamp(max=T - 1)
    tok, st, _ = model.belief.imagine(belief.flatten(0, 1), action[:, idx.clamp(max=T - 1)].flatten(0, 1))
    err = (tok.view(B, T, K, *emb.shape[2:]) - emb[:, tgt_i]).pow(2).mean((-1, -2))  # (B, T, K)
    serr = (st.view(B, T, K, -1) - state[:, tgt_i]).pow(2).mean(-1)
    n = valid.sum() * B
    output["pred_loss"] = (err * valid).sum() / n
    output["pred1_loss"] = (err[:, :, 0] * valid[:, 0]).sum() / (valid[:, 0].sum() * B)
    output["state_loss"] = (serr * valid).sum() / n
    output["sigreg_loss"] = token_sigreg(sigreg, emb, cfg)
    output["loss"] = output["pred_loss"] + cfg.loss.state_weight * output["state_loss"] + lambd * output["sigreg_loss"]
    with torch.no_grad():  # collapse monitors: effective rank of the belief over the batch, within-window spread
        flat = belief.flatten(0, 1).float()
        sv = torch.linalg.svdvals(flat - flat.mean(0))
        p = sv / sv.sum()
        output["belief_rank"] = torch.exp(-(p * torch.log(p + 1e-12)).sum())
        output["belief_time_std"] = belief.float().std(1).mean()
    return output


FORWARD = {"jepa.JEPA": lejepa_forward, "jepa.TokenJEPA": token_forward, "jepa.BeliefJEPA": belief_forward}
LOG_KEYS = ("loss", "pred_loss", "pred1_loss", "state_loss", "sigreg_loss", "belief_rank", "belief_time_std")


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
    A, S = cfg.data.frameskip * train_eps[0]["action"].shape[-1], train_eps[0]["state"].shape[-1]
    with open_dict(cfg):
        m = cfg.model
        if "action_encoder" in m:
            m.action_encoder.input_dim = A
        if "state_encoder" in m:
            m.state_encoder.input_dim = S
        if m._target_ == "jepa.TokenJEPA":
            m.predictor.state_dim = S
        if "belief" in m:
            m.belief.state_dim, m.belief.action_dim = S, A
    forward = FORWARD[cfg.model._target_]
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
    if hasattr(model, "belief") and hasattr(model.belief, "state_mean"):  # MapBelief warps with raw body velocities
        model.belief.state_mean.copy_(stats["state"][0]); model.belief.state_std.copy_(stats["state"][1])
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
                out = forward(model, sigreg, to_cuda(batch), cfg)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clip_val)
            opt.step()
            sched.step()
            step += 1

            if step % cfg.trainer.log_every == 0:
                rec = {f"train/{k}": out[k].item() for k in LOG_KEYS if k in out}
                if step % cfg.trainer.val_every == 0:
                    model.eval()
                    with torch.no_grad(), autocast():
                        vals = [forward(model, sigreg, to_cuda(b), cfg) for _, b in zip(range(20), val)]
                    rec.update({f"val/{k}": sum(v[k].item() for v in vals) / len(vals) for k in LOG_KEYS[1:] if k in vals[0]})
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
