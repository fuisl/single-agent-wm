"""Render one collected episode to mp4 (and optional PNG frames) to eyeball the labels.

Panels: rendered 48x80 depth | ego-centric labels | world top-down map with trajectory.
Ego label colours: black = never seen, grey = seen free, dark blue = visible now (free),
red = obstacle visible now, orange = obstacle seen earlier but out of view (what memory must keep),
dim red outline = obstacle never seen.

    python viz_episode.py ep_00004_env01_trap_collision.npz --png 60,120
"""
import argparse
import subprocess

import cv2
import numpy as np

import scenegen as sg

p = argparse.ArgumentParser()
p.add_argument("npz")
p.add_argument("--out", default=None)
p.add_argument("--png", default="", help="comma-separated step indices to also save as PNG")
args = p.parse_args()
d = np.load(args.npz)
out = args.out or args.npz.replace(".npz", ".mp4")
T = len(d["reward"])
S = 360                                   # panel height


def depth_panel(depth):
    g = ((sg.D_MAX - depth.astype(np.float32)) / (sg.D_MAX - sg.D_MIN) * 255).astype(np.uint8)
    img = cv2.cvtColor(cv2.applyColorMap(g, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (600, S), interpolation=cv2.INTER_NEAREST)


def ego_panel(occ, seen, vis):
    img = np.zeros((64, 64, 3), np.uint8)
    img[seen & ~occ] = (110, 110, 110)
    img[vis & ~occ] = (40, 70, 140)
    img[occ & ~seen] = (70, 20, 20)
    img[occ & seen & ~vis] = (255, 150, 0)
    img[occ & vis] = (230, 30, 30)
    img = cv2.resize(img, (S, S), interpolation=cv2.INTER_NEAREST)
    c = S // 2
    cv2.drawMarker(img, (c, c), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
    return img


SCALE = S / (2 * sg.EXTENT)


def w2px(xy):
    return int((xy[0] + sg.EXTENT) * SCALE), int((sg.EXTENT - xy[1]) * SCALE)


def world_panel(t):
    img = np.full((S, S, 3), 235, np.uint8)
    for cx, cy, hx, hy, yaw in d["boxes"]:
        R = sg.rot2(yaw)
        pts = np.array([w2px(np.array([cx, cy]) + R @ np.array([sx * hx, sy * hy]))
                        for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
        cv2.fillPoly(img, [pts], (90, 90, 100))
    pos = d["pos"][:t + 1, :2]
    cv2.polylines(img, [np.array([w2px(q) for q in pos])], False, (220, 40, 40), 2, cv2.LINE_AA)
    cv2.circle(img, w2px(d["start"]), 5, (40, 120, 220), -1)
    cv2.circle(img, w2px(d["goal"]), 6, (40, 180, 60), -1)
    yaw = float(d["yaw"][t])
    for a in (-sg.HFOV / 2, sg.HFOV / 2):
        tip = d["pos"][t, :2] + sg.D_MAX * np.array([np.cos(yaw + a), np.sin(yaw + a)])
        cv2.line(img, w2px(d["pos"][t, :2]), w2px(tip), (60, 60, 200), 1, cv2.LINE_AA)
    return img


def label(img, text):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


W = 600 + S + S
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{W}x{S}", "-r", "20", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-crf", "23", out], stdin=subprocess.PIPE)
png_steps = {int(s) for s in args.png.split(",") if s}
for t in range(T):
    occ, seen, vis = d["occ"][t], d["seen"][t], d["vis"][t]
    mem = int((occ & seen & ~vis).sum())
    frame = np.concatenate([
        label(depth_panel(d["depth"][t]), f"depth 48x80   t={t / 20:5.2f}s"),
        label(ego_panel(occ, seen, vis), f"ego labels  memory cells={mem}"),
        label(world_panel(t), f"{d['mode']} -> {d['outcome']}"),
    ], 1)
    ff.stdin.write(frame.tobytes())
    if t in png_steps:
        cv2.imwrite(out.replace(".mp4", f"_t{t:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
ff.stdin.close(); ff.wait()
print("wrote", out)
