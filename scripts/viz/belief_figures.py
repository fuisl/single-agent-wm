"""What the R-LeWM belief does over time (runs/kin_v1/m2..m4), into runs/kin_v1/viz/belief/.

    source env.sh
    $PY_MAIN scripts/viz/belief_figures.py traj --run_dirs runs/kin_v1/m2 runs/kin_v1/m3 runs/kin_v1/m4
    $PY_MAIN scripts/viz/belief_figures.py map --run_dir runs/kin_v1/m4 --memtest data/memtest_v1/sc_000_gap040_seen.npz
    $PY_MAIN scripts/viz/belief_figures.py map --run_dir runs/kin_v1/m4 --episode ep_00060_kin_trap_goal.npz

  traj  belief_traj_<ep>.png: 2-D PCA of the belief b_t (features_belief) over one held-out trap episode, coloured
        by time, with markers where obstacles leave the view (many cells become "remembered") and re-enter it;
        below, the counts of visible and remembered obstacle cells in the ego map
  map   (M4) MP4 + key-frame PNGs of the latent map: its 32 channels as RGB via PCA (fitted over the rollout) on
        the 16x16 grid next to the ego occupancy truth at the same scale (9.6 m, drone at the centre facing up),
        the onboard depth and the chase camera. A remembered wall should stay in the map while out of view and
        rotate with the drone.
"""
import argparse
import glob
import os

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lewmlib
import scenegen as sg
from vizlib import DEPTH_CMAP, chase_cam, ego_truth, even, model_label

p = argparse.ArgumentParser()
p.add_argument("what", choices=["traj", "map"])
p.add_argument("--run_dirs", nargs="*", default=[os.path.join(lewmlib.ROOT, f"runs/kin_v1/m{i}") for i in (2, 3, 4)])
p.add_argument("--run_dir", default=os.path.join(lewmlib.ROOT, "runs/kin_v1/m4"))
p.add_argument("--data", default=os.path.join(lewmlib.ROOT, "data/kin_v1"))
p.add_argument("--episode", default=None, help="kin_v1 episode file name (default: first val trap episode found)")
p.add_argument("--memtest", default=None, help="a memtest rollout npz (map mode)")
p.add_argument("--device", default="cpu")
p.add_argument("--fps", type=int, default=6)
p.add_argument("--out", default=os.path.join(lewmlib.ROOT, "runs/kin_v1/viz/belief"))
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)


def default_episode(feat_dirs):
    have = None
    for d in feat_dirs:
        s = {os.path.basename(f) for f in glob.glob(os.path.join(d, "*_trap_*.npz")) if str(np.load(f)["split"]) == "val"}
        have = s if have is None else have & s
    assert have, f"no val trap episode with features in {feat_dirs}"
    return sorted(have)[0]


def view_events(d):
    """Visible / remembered obstacle cells per step, and the steps where obstacles leave the view (remembered cells
    up while visible cells go down, over 1 s) or come back into it (remembered down while visible up)."""
    occ, seen, vis = d["occ"], d["seen"], d["vis"]
    n_vis = (occ & vis).sum((1, 2))
    n_mem = (occ & seen & ~vis).sum((1, 2))
    w = 20  # obstacles leave / re-enter the view gradually while the drone turns
    dm, dv = np.zeros(len(n_mem)), np.zeros(len(n_vis))
    dm[w:], dv[w:] = n_mem[w:] - n_mem[:-w], n_vis[w:] - n_vis[:-w]
    score = np.minimum(np.abs(dm), np.abs(dv)) * (np.sign(dm) != np.sign(dv))  # opposite directions only
    thr = max(np.percentile(score, 95), 8)
    leave, enter = [], []
    for t in range(w, len(score)):
        if score[t] < thr or score[t] != score[max(t - w, 0):t + w].max():
            continue
        lst = leave if dm[t] > 0 else enter
        if not lst or t - lst[-1] > w:
            lst.append(t - w // 2)
    return n_vis, n_mem, leave, enter


# ---------------------------------------------------------------- belief trajectories
def traj():
    runs = [r for r in args.run_dirs if os.path.isdir(os.path.join(r, "features_belief"))]
    if not runs:
        print("note: no features_belief in", args.run_dirs); return
    name = args.episode or default_episode([os.path.join(r, "features_belief") for r in runs])
    d = np.load(os.path.join(args.data, name))
    n_vis, n_mem, leave, enter = view_events(d)
    T = len(n_vis)
    fig = plt.figure(figsize=(5.2 * len(runs), 7.2))
    gs = fig.add_gridspec(3, len(runs), hspace=0.45)
    for i, r in enumerate(runs):
        b = np.load(os.path.join(r, "features_belief", name))["feat"].astype(np.float32)
        b = b - b.mean(0)
        _, _, Vt = np.linalg.svd(b[::2], full_matrices=False)
        P = b @ Vt[:2].T
        a = fig.add_subplot(gs[:2, i])
        a.plot(P[:, 0], P[:, 1], color=(0.8, 0.8, 0.8), lw=0.6, zorder=0)
        sc = a.scatter(P[:, 0], P[:, 1], c=np.arange(T) / 20, cmap="viridis", s=5, zorder=1)
        a.scatter(P[leave, 0], P[leave, 1], marker="^", s=90, color="#D55E00", edgecolor="white", zorder=2,
                  label="obstacles leave the view")
        a.scatter(P[enter, 0], P[enter, 1], marker="o", s=70, color="#0072B2", edgecolor="white", zorder=2,
                  label="obstacles re-enter the view")
        a.set_title(model_label(os.path.basename(os.path.normpath(r))), fontsize=12)
        a.set_xticks([]); a.set_yticks([]); a.set_xlabel("belief PC 1"); a.set_ylabel("belief PC 2")
        if i == 0:
            a.legend(frameon=False, fontsize=9, loc="best")
        if i == len(runs) - 1:
            fig.colorbar(sc, ax=a, label="time (s)", fraction=0.046)
    a = fig.add_subplot(gs[2, :])
    t = np.arange(T) / 20
    a.plot(t, n_vis, color="#D55E00", label="obstacle cells visible now")
    a.plot(t, n_mem, color="#E69F00", label="obstacle cells seen, now out of view (to remember)")
    for x in leave:
        a.axvline(x / 20, color="#D55E00", ls=":", lw=1)
    for x in enter:
        a.axvline(x / 20, color="#0072B2", ls=":", lw=1)
    a.set_xlabel("time (s)"); a.set_ylabel("ego-map cells")
    a.legend(frameon=False, fontsize=9, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Belief trajectory over {name.replace('.npz', '')} (held-out trap episode)", fontsize=13)
    f = os.path.join(args.out, f"belief_traj_{name.replace('.npz', '')}.png")
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", f, flush=True)


# ---------------------------------------------------------------- M4 latent map video
def memtest_truth(d):
    """Ego occupancy truth per step of a memtest rollout (W present per the variant's schedule)."""
    variant = os.path.basename(args.memtest).rsplit("_", 1)[1].replace(".npz", "")
    phase = d["phase"]
    with_w = np.concatenate([d["pillars"], d["wall"][None]]).astype(np.float64)
    seen_map, out = sg.SeenMap(), {"occ": [], "seen": [], "vis": [], "boxes": []}
    for t in range(len(phase)):
        present = {"seen": True, "absent": False, "novel": phase[t] >= 2, "removed": phase[t] < 2}[variant]
        boxes = with_w if present else d["pillars"].astype(np.float64)
        _, occ, seen, vis = sg.labels(d["pos"][t].astype(np.float64), float(d["yaw"][t]), boxes, seen_map)
        for k, v in zip(["occ", "seen", "vis", "boxes"], [occ, seen, vis, boxes]):
            out[k].append(v)
    return {k: (np.array(v) if k != "boxes" else v) for k, v in out.items()}, variant


def map_video():
    M = lewmlib.Model(args.run_dir, args.device)
    assert M.is_map, f"{args.run_dir} is not an M4 (MapBelief) model"
    if args.memtest:
        d = dict(np.load(args.memtest))
        truth, variant = memtest_truth(d)
        tag = os.path.basename(args.memtest).replace(".npz", "")
        boxes_at = lambda t: truth["boxes"][t]
        phase = d["phase"]
    else:
        name = args.episode or default_episode([os.path.join(args.run_dir, "features_belief")])
        d = dict(np.load(os.path.join(args.data, name)))
        truth = {k: d[k] for k in ("occ", "seen", "vis")}
        tag, variant, phase = name.replace(".npz", ""), None, None
        boxes_at = lambda t: d["boxes"].astype(np.float64)
    fr, dp, st, bl = M.frames(d)
    z, s, a = M.embed(dp, st, bl)
    _, maps = M.observe(z, s, a)
    m = maps.permute(0, 2, 3, 1).reshape(-1, maps.shape[1]).float()
    m = m - m.mean(0)
    _, _, V = torch.pca_lowrank(m, q=3, center=False)
    pc = (m @ V[:, :3]).reshape(len(fr), maps.shape[2], maps.shape[3], 3).cpu().numpy()
    lo, hi = np.percentile(pc, 2, axis=(0, 1, 2)), np.percentile(pc, 98, axis=(0, 1, 2))
    rgb = np.clip((pc - lo) / (hi - lo + 1e-9), 0, 1)
    energy = maps.float().pow(2).mean(1).cpu().numpy()
    names = ["facing W", "turning away", "gap: W out of view", "turning back", "hover"]

    fig, ax = plt.subplots(1, 5, figsize=(18, 3.9), dpi=90, gridspec_kw={"width_ratios": [1.6, 1.25, 1, 1, 1]})
    title = fig.suptitle("", fontsize=13, x=0.01, ha="left")
    writer = imageio.get_writer(os.path.join(args.out, f"m4map_{tag}.mp4"), fps=args.fps, macro_block_size=1, quality=7)
    keys = {0, len(fr) // 2, len(fr) - 1}
    if phase is not None:
        keys |= {int(np.argmax(phase[fr] == 2)), int(np.argmax(phase[fr] >= 3)) - 1}
    for j, t in enumerate(fr):
        for x in ax:
            x.clear(); x.set_xticks([]); x.set_yticks([])
        img, proj = chase_cam(d["pos"][t].astype(np.float64), float(d["yaw"][t]), boxes_at(t))
        ax[0].imshow(img, interpolation="bilinear"); ax[0].set_title("chase camera", fontsize=10)
        ax[1].imshow(d["depth"][t].astype(np.float32), cmap=DEPTH_CMAP, vmin=sg.D_MIN, vmax=sg.D_MAX)
        ax[1].set_title("onboard depth", fontsize=10)
        ax[2].imshow(ego_truth(truth["occ"][t], truth["seen"][t], truth["vis"][t]), interpolation="nearest")
        ax[2].set_title("ego truth (orange = remembered)", fontsize=10)
        ax[3].imshow(rgb[j], interpolation="nearest"); ax[3].set_title("M4 latent map (PCA RGB)", fontsize=10)
        ax[4].imshow(energy[j], cmap="magma", vmin=0, vmax=np.percentile(energy, 99), interpolation="nearest")
        ax[4].set_title("map energy per cell", fontsize=10)
        for x, c in [(ax[2], 31.5), (ax[3], 7.5), (ax[4], 7.5)]:
            x.plot(c, c, marker="^", ms=8, color="white", mec="black")
        ph = f"    phase: {names[phase[t]]}" if phase is not None else ""
        title.set_text(f"{model_label('m4')}: {tag}{f' ({variant})' if variant else ''}    t = {t / 20:5.2f} s{ph}"
                       "    (maps: 9.6 m x 9.6 m, drone at centre facing up)")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(even(frame))
        if j in keys:
            imageio.imwrite(os.path.join(args.out, f"m4map_{tag}_f{j:03d}.png"), frame)
    writer.close(); plt.close(fig)
    print("wrote", os.path.join(args.out, f"m4map_{tag}.mp4"), flush=True)


traj() if args.what == "traj" else map_video()
