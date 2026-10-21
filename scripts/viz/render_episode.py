"""Render a collected episode with what the LeWM latents (and the probes) make of it, for talks and debugging.

    source env.sh
    $PY_MAIN scripts/viz/render_episode.py                                   # 2 val trap episodes
    $PY_MAIN scripts/viz/render_episode.py --episodes ep_00060_kin_trap_goal.npz --steps 200:500 --png 300,420

Panels (missing inputs are skipped with a note):
  world      top-down map: boxes, trajectory, goal, field of view (ray-cast), obstacle cells visible now (red)
             and seen before but out of view (orange) = what a memory has to keep
  chase cam  pseudo-3D render of the box world from behind the drone (scenegen ray casting, Lambert shading)
  depth      onboard 48x80 depth | decoded from LeWM's final emb (heads.pt) | from the ViT CLS (heads_cls.pt)
             | from the 6x10 patch tokens (heads_patch.pt) | patch-token PCA as RGB ("what the encoder sees")
  ego maps   64x64 ego occupancy truth (probe.py colours) and probe predictions from <run>/probe*/trap_preds.npz
Writes <out>/<episode>.mp4 (+ PNG key frames).
"""
import argparse
import glob
import os
import sys

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import scenegen as sg

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "lewm"))
from utils import depth_to_pixels, encode_steps, is_val, load_ckpt, load_episode  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default=os.path.join(ROOT, "data/kin_v1"))
p.add_argument("--run_dir", default=os.path.join(ROOT, "runs/kin_v1/lewm"))
p.add_argument("--probe_root", default=os.path.join(ROOT, "runs/kin_v1"), help="searched for probe*/trap_preds.npz")
p.add_argument("--probe_sets", nargs="*", default=["wm", "lewm_emb", "lewm_patch", "frame"])
p.add_argument("--episodes", nargs="*", default=None, help="episode file names (default: 2 val trap episodes)")
p.add_argument("--steps", default=None, help="start:end step range")
p.add_argument("--every", type=int, default=2, help="render every n-th step (20 Hz data)")
p.add_argument("--fps", type=int, default=10)
p.add_argument("--png", default="", help="comma-separated steps also saved as PNG")
p.add_argument("--out", default=os.path.join(ROOT, "runs/kin_v1/viz/episodes"))
p.add_argument("--device", default="cuda", help="cpu works too (slower), e.g. while the GPU is busy training")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
from vizlib import DEPTH_CMAP, box_corners, chase_cam, ego_truth, even, label  # noqa: E402

# ---------------------------------------------------------------- LeWM read-outs
@torch.no_grad()
def lewm_panels(e):
    """Decoded depth images (T, 48, 80) in metres per available decoder, and patch PCA RGB (T, 48, 80, 3)."""
    ck = os.path.join(args.run_dir, "lewm_object.ckpt")
    if not os.path.exists(ck):
        print(f"note: no {ck}, skipping LeWM panels")
        return {}, None
    ckpt = load_ckpt(ck, args.device)
    model, stats = ckpt["model"], ckpt["stats"]
    hs = torch.cat([model.encoder(depth_to_pixels(e["depth"][c:c + 256]).to(args.device)).last_hidden_state.float()
                    for c in range(0, len(e["depth"]), 256)])
    feats = {"emb": encode_steps(model, stats, e["depth"], e["state"]).float(), "cls": hs[:, 0],
             "patch": hs[:, 1:].reshape(len(hs), 6, 10, -1).permute(0, 3, 1, 2)}
    out = {}
    for rep, name in [("emb", "heads.pt"), ("cls", "heads_cls.pt"), ("patch", "heads_patch.pt")]:
        f = os.path.join(args.run_dir, name)
        if not os.path.exists(f):
            print(f"note: no {f}, skipping decoded-from-{rep} panel")
            continue
        dec = torch.load(f, map_location=args.device, weights_only=False)["decoder"].eval()
        x = feats[rep]
        rec = torch.cat([dec(x[c:c + 256]).float() for c in range(0, len(x), 256)]).cpu().numpy()
        out[rep] = sg.D_MIN + rec * (sg.D_MAX - sg.D_MIN)
    tok = hs[:, 1:].reshape(-1, hs.shape[-1])
    tok = tok - tok.mean(0)
    _, _, V = torch.pca_lowrank(tok, q=3, center=False)
    pc = (tok @ V[:, :3]).reshape(len(hs), 6, 10, 3).cpu().numpy()
    lo, hi = np.percentile(pc, 2, axis=(0, 1, 2)), np.percentile(pc, 98, axis=(0, 1, 2))
    pc = np.clip((pc - lo) / (hi - lo), 0, 1)
    pca = np.repeat(np.repeat(pc, 8, 1), 8, 2)                      # 6x10 patches of 8x8 pixels -> 48x80
    return out, pca


def probe_maps(name):
    """{feature set: (T, 64, 64)} from every probe*/trap_preds.npz under probe_root (first occurrence wins)."""
    maps = {}
    for f in sorted(glob.glob(os.path.join(args.probe_root, "probe*", "trap_preds.npz"))):
        z = np.load(f)
        for k in z.files:
            s, ep = k.split("/", 1)
            if ep == name and s in args.probe_sets and s not in maps:
                maps[s] = z[k].astype(np.float32)
    missing = [s for s in args.probe_sets if s not in maps]
    if missing:
        print(f"note: no probe maps for {missing} on {name}")
    return {s: maps[s] for s in args.probe_sets if s in maps}


# ---------------------------------------------------------------- world views
GX = -sg.EXTENT + (np.arange(sg.SEEN_N) + 0.5) * sg.SEEN_RES
GRID = np.stack(np.meshgrid(GX, GX, indexing="ij"), -1)


def world_masks(d):
    """Per step, world-frame obstacle cells visible now and seen-before-but-out-of-view (SEEN_N x SEEN_N)."""
    boxes = d["boxes"].astype(np.float64)
    occ = sg.min_dist(GRID.reshape(-1, 2), boxes).reshape(sg.SEEN_N, sg.SEEN_N) <= 0
    seen = np.zeros_like(occ)
    w = int(np.ceil((sg.D_MAX + 0.3) / sg.SEEN_RES))
    vis_all, mem_all, rays = [], [], []
    for pos, yaw in zip(d["pos"].astype(np.float64), d["yaw"].astype(np.float64)):
        dirs = np.stack([np.cos(yaw + sg.RAY_ANG), np.sin(yaw + sg.RAY_ANG)], -1)
        rr = sg.raycast2d(pos[:2], dirs, boxes, sg.D_MAX)
        vis = np.zeros_like(occ)
        ci, cj = ((pos[:2] + sg.EXTENT) / sg.SEEN_RES).astype(int)
        i0, i1, j0, j1 = max(ci - w, 0), min(ci + w + 1, sg.SEEN_N), max(cj - w, 0), min(cj + w + 1, sg.SEEN_N)
        if i0 < i1 and j0 < j1:
            vis[i0:i1, j0:j1] = sg.visibility(GRID[i0:i1, j0:j1], pos, yaw, rr)
        vis &= occ
        mem_all.append(seen & ~vis)
        vis_all.append(vis)
        seen = seen | vis
        rays.append(pos[:2] + dirs * rr[:, None])
    return np.array(vis_all), np.array(mem_all), np.array(rays)


# ---------------------------------------------------------------- frame drawing
def render(name):
    path = os.path.join(args.data, name)
    d = np.load(path)
    e = load_episode(path)
    T = len(d["reward"])
    lo, hi = (int(x) for x in args.steps.split(":")) if args.steps else (0, T)
    hi = min(hi, T)
    dec, pca = lewm_panels(e)
    maps = probe_maps(name)
    vis_w, mem_w, rays = world_masks(d)
    boxes = d["boxes"].astype(np.float64)
    pos, yaw = d["pos"].astype(np.float64), d["yaw"].astype(np.float64)
    img_panels = [("onboard depth (truth)", lambda t: d["depth"][t].astype(np.float32), "depth")]
    for rep, lab in [("emb", "decoded: LeWM final emb"), ("cls", "decoded: ViT CLS"), ("patch", "decoded: patch tokens")]:
        if rep in dec:
            img_panels.append((lab, (lambda r: lambda t: dec[r][t])(rep), "depth"))
    if pca is not None:
        img_panels.append(("patch-token PCA (RGB)", lambda t: pca[t], "rgb"))
    ego = [("ego truth", None)] + [(label(s), s) for s in maps]

    fig = plt.figure(figsize=(16, 9), dpi=90, facecolor="white")
    gs = fig.add_gridspec(6, 12, left=0.02, right=0.99, top=0.93, bottom=0.03, wspace=0.35, hspace=0.6)
    ax_w = fig.add_subplot(gs[:, :5])
    ax_c = fig.add_subplot(gs[:3, 5:9])
    n_img = len(img_panels)
    ax_i = [fig.add_subplot(gs[3 + i // 2, 5 + 2 * (i % 2): 7 + 2 * (i % 2)]) for i in range(min(n_img, 6))]
    ge = gs[:, 9:].subgridspec(3, 2, hspace=0.35, wspace=0.15)
    ax_e = [fig.add_subplot(ge[i // 2, i % 2]) for i in range(min(len(ego), 6))]
    title = fig.suptitle("", fontsize=14, x=0.02, ha="left")
    writer = imageio.get_writer(os.path.join(args.out, name.replace(".npz", ".mp4")), fps=args.fps,
                                macro_block_size=1, quality=7)
    pngs = {int(x) for x in args.png.split(",") if x}
    for t in range(lo, hi):
        if (t - lo) % args.every and t not in pngs:
            continue
        for a in [ax_w, ax_c, *ax_i, *ax_e]:
            a.clear(); a.set_xticks([]); a.set_yticks([])
        # world
        for b in boxes:
            ax_w.add_patch(Polygon(box_corners(b), closed=True, fc=(0.55, 0.55, 0.55), ec="none"))
        ax_w.add_patch(Polygon(np.vstack([pos[t, :2], rays[t]]), closed=True, fc=(0.2, 0.45, 0.85, 0.18), ec="none"))
        ext = [-sg.EXTENT, sg.EXTENT, -sg.EXTENT, sg.EXTENT]
        ov = np.zeros((sg.SEEN_N, sg.SEEN_N, 4))
        ov[mem_w[t]] = (1, 0.6, 0, 1)
        ov[vis_w[t]] = (0.9, 0.1, 0.1, 1)
        ax_w.imshow(ov.transpose(1, 0, 2), origin="lower", extent=ext, interpolation="nearest", zorder=3)
        ax_w.plot(pos[:t + 1, 0], pos[:t + 1, 1], color=(0.1, 0.1, 0.1), lw=1.2)
        ax_w.plot(*d["goal"], marker="*", ms=16, color=(0.1, 0.6, 0.2))
        ax_w.plot(*d["start"], marker="o", ms=7, color=(0.3, 0.3, 0.3))
        c, s = np.cos(yaw[t]), np.sin(yaw[t])
        tri = pos[t, :2] + np.array([[0.75 * c, 0.75 * s], [-0.4 * c - 0.35 * s, -0.4 * s + 0.35 * c],
                                     [-0.4 * c + 0.35 * s, -0.4 * s - 0.35 * c]])
        ax_w.add_patch(Polygon(tri, closed=True, fc="black"))
        ax_w.set_xlim(-sg.EXTENT, sg.EXTENT); ax_w.set_ylim(-sg.EXTENT, sg.EXTENT); ax_w.set_aspect("equal")
        ax_w.set_title("world (top-down): red = obstacle visible now, orange = seen, now out of view", fontsize=10)
        # chase cam
        img, proj = chase_cam(pos[t], yaw[t], boxes)
        ax_c.imshow(img, interpolation="bilinear")
        tr = [proj(P) for P in pos[max(t - 100, 0):t + 1]]
        tr = np.array([uv for uv in tr if uv is not None])
        if len(tr) > 1:
            ax_c.plot(tr[:, 0], tr[:, 1], color=(0.1, 0.1, 0.1), lw=1.5)
        for P, mk, col in [(pos[t], "o", "black"), (np.array([*d["goal"], sg.Z_GOAL]), "*", (0.1, 0.6, 0.2))]:
            uv = proj(P)
            if uv is not None and 0 <= uv[0] < img.shape[1] and 0 <= uv[1] < img.shape[0]:
                ax_c.plot(*uv, marker=mk, ms=10, color=col, mec="white")
        ax_c.set_xlim(0, img.shape[1]); ax_c.set_ylim(img.shape[0], 0)
        ax_c.set_title("chase camera (rendered from the box world, last 5 s of path)", fontsize=10)
        # depth / decodes / PCA
        for a, (lab, fn, kind) in zip(ax_i, img_panels):
            x = fn(t)
            a.imshow(x, cmap=DEPTH_CMAP, vmin=sg.D_MIN, vmax=sg.D_MAX) if kind == "depth" else a.imshow(x)
            a.set_title(lab, fontsize=9)
        # ego maps
        for a, (lab, s) in zip(ax_e, ego):
            if s is None:
                a.imshow(ego_truth(d["occ"][t], d["seen"][t], d["vis"][t]), interpolation="nearest")
            else:
                a.imshow(maps[s][t], vmin=0, vmax=1, cmap="magma", interpolation="nearest")
            a.plot(31.5, 31.5, marker="^", ms=6, color="white")
            a.set_title(lab, fontsize=9)
        title.set_text(f"{name.replace('.npz', '')}    t = {t / 20:5.2f} s    "
                       f"(ego maps: drone at centre facing up, 9.6 m x 9.6 m)")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        if (t - lo) % args.every == 0:
            writer.append_data(even(frame))
        if t in pngs:
            imageio.imwrite(os.path.join(args.out, name.replace(".npz", f"_t{t:04d}.png")), frame)
    writer.close()
    plt.close(fig)
    print(f"wrote {os.path.join(args.out, name.replace('.npz', '.mp4'))}", flush=True)


names = args.episodes or [os.path.basename(f) for f in sorted(glob.glob(os.path.join(args.data, "ep_*_trap_*.npz")))
                          if is_val(f)][:2]
for n in names:
    render(n)
