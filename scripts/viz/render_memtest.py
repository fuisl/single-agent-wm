"""The twin look-away memory test as a video: the 4 variants of one scenario side by side, with each model's
latent prediction error over time (uav_env/kin_memtest.py scenarios, lewm/memtest.py errors).

    source env.sh
    $PY_MAIN scripts/viz/render_memtest.py --scenario 0 --gap 40 --run_dirs runs/kin_v1/lewm runs/kin_v1/m2
    $PY_MAIN scripts/viz/render_memtest.py --data runs/kin_v1/viz/mt_small --run_dirs <smoke dirs>   # test

Top: chase camera (wall W in blue while present, dashed ghost outline while absent) and onboard depth for
  seen     W there all along                 novel    W appears while the drone looks away
  removed  W removed while the drone looks away   absent   W never there
"seen" and "novel" (and "removed" / "absent") show identical frames from the gap on; only the past differs.
Bottom: per model, the one-step latent prediction error per frame (normalised by the model's target variance,
from its memtest.json when present), re-entry frames shaded. Memory = novel above seen at re-entry.
Writes <out>/sc<id>_gap<g>.mp4 and PNG key frames (end of gap, re-entry, hover).
"""
import argparse
import os

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lewmlib
import scenegen as sg
from vizlib import DEPTH_CMAP, chase_cam, even, model_label, prism_edges

p = argparse.ArgumentParser()
p.add_argument("--data", default=os.path.join(lewmlib.ROOT, "data/memtest_v1"))
p.add_argument("--scenario", type=int, default=0)
p.add_argument("--gap", type=int, default=40)
p.add_argument("--run_dirs", nargs="*", default=[])
p.add_argument("--device", default="cpu")
p.add_argument("--every", type=int, default=2, help="render every n-th env step (20 Hz)")
p.add_argument("--fps", type=int, default=10)
p.add_argument("--min_wall_px", type=int, default=200)
p.add_argument("--out", default=os.path.join(lewmlib.ROOT, "runs/kin_v1/viz/memtest"))
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
VARIANTS = [("seen", "W there all along", "#0072B2"), ("novel", "W appears during the gap", "#E69F00"),
            ("removed", "W removed during the gap", "#D55E00"), ("absent", "W never there", "#009E73")]
PHASES = ["facing W", "turning away", "gap: W out of view", "turning back", "hover"]

stem = os.path.join(args.data, f"sc_{args.scenario:03d}_gap{args.gap:03d}")
runs = {v: dict(np.load(f"{stem}_{v}.npz")) for v, _, _ in VARIANTS}
seen = runs["seen"]
wall, pillars = seen["wall"].astype(np.float64), seen["pillars"].astype(np.float64)
phase = seen["phase"]
T = len(phase)
j0 = int(np.argmax(phase >= 3))  # first turn-back step


def present(v, t):
    return {"seen": True, "absent": False, "novel": phase[t] >= 2, "removed": phase[t] < 2}[v]


# ---------------------------------------------------------------- model errors (frame level)
curves = []
for rd in args.run_dirs:
    M = lewmlib.Model(rd, args.device)
    fr, *_ = M.frames(seen)
    zs, es = {}, {}
    for v, _, _ in VARIANTS:
        _, dp, st, bl = M.frames(runs[v])
        z, s, a = M.embed(dp, st, bl)
        zs[v], es[v] = z, M.errors(z, s, a)
    var = M.target_var or float(zs["seen"].flatten(0, 1).var(0).mean())
    curves.append((model_label(os.path.basename(os.path.normpath(rd))), fr / 20.0, {v: e / var for v, e in es.items()}))
    print(f"errors: {rd} ({M.kind})", flush=True)
fr_all = np.arange(0, T, 5)
re_fr = fr_all[(phase[fr_all] >= 3) & (seen["wall_px"][fr_all] > args.min_wall_px)][:4]

# ---------------------------------------------------------------- figure
nm = max(len(curves), 1)
fig = plt.figure(figsize=(16, 9), dpi=90, facecolor="white")
gs = fig.add_gridspec(9, 4 * nm if curves else 4, left=0.04, right=0.99, top=0.88, bottom=0.07, hspace=0.9, wspace=0.6)
ax_c, ax_d = [], []
for i in range(4):
    sl = slice(i * nm, (i + 1) * nm) if curves else slice(i, i + 1)
    ax_c.append(fig.add_subplot(gs[0:4, sl]))
    ax_d.append(fig.add_subplot(gs[4:6, sl]))
ax_e = [fig.add_subplot(gs[6:9, 4 * j:4 * (j + 1)]) for j in range(len(curves))]
fig.suptitle("Twin look-away test   seen vs novel: identical frames at re-entry, different past", fontsize=16,
             x=0.04, ha="left", y=0.975)
sub = fig.text(0.04, 0.925, "", fontsize=12)
lines = []
for a, (name, x, ev) in zip(ax_e, curves):
    for v, _, col in VARIANTS:
        a.plot(x, ev[v], color=col, lw=2 if v in ("seen", "novel") else 1.3,
               ls="-" if v in ("seen", "novel") else "--", label=v)
    for f in re_fr:
        a.axvspan((f - 2.5) / 20, (f + 2.5) / 20, color=(0.85, 0.85, 0.85), zorder=0)
    a.axvspan(np.argmax(phase == 2) / 20, j0 / 20, color=(0.95, 0.93, 0.85), zorder=0)
    a.set_ylim(0, np.nanpercentile(np.concatenate(list(ev.values())), 99.5) * 1.15 + 1e-9); a.set_xlim(0, T / 20)
    a.set_title(name, fontsize=11); a.set_xlabel("time (s)   tan = gap, grey = re-entry frames", fontsize=9)
    a.set_ylabel("latent pred. error", fontsize=9)
    a.spines[["top", "right"]].set_visible(False)
    lines.append(a.axvline(0, color="black", lw=1))
if ax_e:
    ax_e[0].legend(fontsize=8, frameon=False, ncol=2, loc="upper left")

writer = imageio.get_writer(f"{os.path.join(args.out, os.path.basename(stem).replace('sc_', 'sc'))}.mp4",
                            fps=args.fps, macro_block_size=1, quality=7)
keys = {max(j0 - 2, 0): "end_of_gap", int(re_fr[0]) if len(re_fr) else j0 + 20: "reentry", T - 2: "hover"}
for t in range(T):
    if t % args.every and t not in keys:
        continue
    for i, (v, desc, col) in enumerate(VARIANTS):
        r = runs[v]
        boxes = np.concatenate([pillars, wall[None]]) if present(v, t) else pillars
        img, proj = chase_cam(r["pos"][t].astype(np.float64), float(r["yaw"][t]), boxes,
                              highlight=len(pillars) if present(v, t) else None)
        a = ax_c[i]; a.clear(); a.set_xticks([]); a.set_yticks([])
        a.imshow(img, interpolation="bilinear")
        if not present(v, t):
            for P0, P1 in prism_edges(wall):
                u0, u1 = proj(P0), proj(P1)
                if u0 is not None and u1 is not None:
                    a.plot([u0[0], u1[0]], [u0[1], u1[1]], color=(0.2, 0.42, 0.85), lw=1, ls=(0, (3, 3)), alpha=0.8)
        uv = proj(r["pos"][t].astype(np.float64))
        if uv is not None:
            a.plot(*uv, marker="o", ms=8, color="black", mec="white")
        a.set_xlim(0, img.shape[1]); a.set_ylim(img.shape[0], 0)
        a.set_title(f"{v}: {desc}", fontsize=11, color=col, fontweight="bold")
        b = ax_d[i]; b.clear(); b.set_xticks([]); b.set_yticks([])
        b.imshow(r["depth"][t].astype(np.float32), cmap=DEPTH_CMAP, vmin=sg.D_MIN, vmax=sg.D_MAX)
        b.set_title("onboard depth", fontsize=9)
    for ln in lines:
        ln.set_xdata([t / 20, t / 20])
    sub.set_text(f"scenario {args.scenario}, gap {args.gap / 20:.1f} s    t = {t / 20:5.2f} s    phase: {PHASES[phase[t]]}"
                 f"    (blue = wall W, dashed = where W would be)")
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    if t % args.every == 0:
        writer.append_data(even(frame))
    if t in keys:
        imageio.imwrite(os.path.join(args.out, f"sc{args.scenario:03d}_gap{args.gap:03d}_{keys[t]}.png"), frame)
writer.close()
print("wrote", args.out, flush=True)
