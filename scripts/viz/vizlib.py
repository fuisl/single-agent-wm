"""Shared drawing helpers for scripts/viz: chase-camera ray casting of the box world, box outlines, ego-map
colouring and display names of feature sets / models."""
import numpy as np

import scenegen as sg

LABELS = {"wm": "DreamerV3 (reference)", "frame": "current frame only (CNN)", "lewm_emb": "LeWM final emb",
          "lewm_ctx": "LeWM 3-frame ctx", "lewm_cls": "LeWM CLS", "lewm_patch": "LeWM patch tokens",
          "lewm_pred": "LeWM predictor", "lewm_pred_h": "LeWM predictor (hidden)"}
DEPTH_CMAP = "turbo_r"


def label(k):
    """Display name of a probe feature set; m<i>_<part> names come from scripts/lewm_memory.sh evaluate()."""
    if k in LABELS:
        return LABELS[k]
    m, _, part = k.partition("_")
    return f"{m.upper()} {part}" if m[:1] == "m" and m[1:].isdigit() else k


MODEL_LABELS = {"lewm": "LeWM (official, 3-frame window)", "m0": "M0 pixels-only + proprio",
                "m1": "M1 + spatial tokens", "m2": "M2 + recurrent belief", "m3": "M3 + memory-rich data",
                "m4": "M4 + ego-motion map", "wm": "DreamerV3 (reference)"}


def model_label(run_name):
    """Display name of a run directory (runs/kin_v1/<name>; smoke_<name> for smoke tests)."""
    n = run_name.replace("smoke_", "")
    return MODEL_LABELS.get(n, run_name)


def box_corners(b):
    cx, cy, hx, hy, yaw = b
    R = sg.rot2(yaw)
    return np.array([np.array([cx, cy]) + R @ np.array([sx * hx, sy * hy]) for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])


def prism_hits(o, d, boxes):
    """First hit of rays o + t d (d (R, 3)) with the boxes as prisms of height OBS_H.
    Returns t (R,), box index (R,), top-face flag (R,); t = inf where nothing is hit."""
    if len(boxes) == 0:
        return np.full(len(d), np.inf), np.zeros(len(d), int), np.zeros(len(d), bool)
    rel = o[:2] - boxes[:, :2]                                              # (B, 2)
    c, s = np.cos(boxes[:, 4]), np.sin(boxes[:, 4])
    ox, oy = c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]
    dx = c * d[:, None, 0] + s * d[:, None, 1]
    dy = -s * d[:, None, 0] + c * d[:, None, 1]
    dx = np.where(np.abs(dx) < 1e-9, 1e-9, dx); dy = np.where(np.abs(dy) < 1e-9, 1e-9, dy)
    tx1, tx2 = (-boxes[:, 2] - ox) / dx, (boxes[:, 2] - ox) / dx
    ty1, ty2 = (-boxes[:, 3] - oy) / dy, (boxes[:, 3] - oy) / dy
    t0 = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))           # (R, B) 2D entry / exit
    t1 = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    ok = t1 >= np.maximum(t0, 0)
    z0 = o[2] + t0 * d[:, None, 2]
    side = ok & (t0 > 0) & (z0 >= 0) & (z0 <= sg.OBS_H)
    t_top = np.where(d[:, None, 2] < -1e-9, (sg.OBS_H - o[2]) / np.minimum(d[:, None, 2], -1e-9), np.inf)
    top = ok & (z0 > sg.OBS_H) & (t_top >= t0) & (t_top <= t1)
    t = np.where(side, t0, np.where(top, t_top, np.inf))
    k = t.argmin(-1)
    r = np.arange(len(d))
    return t[r, k], k, top[r, k]


def chase_cam(pos, yaw, boxes, res=(320, 192), hfov=np.deg2rad(90), back=3.5, up=4.0, pitch=np.deg2rad(42),
              highlight=None):
    """Shaded pseudo-3D view from behind and above the drone; returns (H, W, 3) image and a pixel projector.
    highlight: index of a box drawn in blue instead of sand."""
    W, H = res
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    cam = np.array([pos[0] - back * cy, pos[1] - back * sy, pos[2] + up])
    fx = (W / 2) / np.tan(hfov / 2)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = (np.stack([np.ones_like(u), -(u - W / 2) / fx, -(v - H / 2) / fx], -1).reshape(-1, 3)) @ R.T
    t_box, k, is_top = prism_hits(cam, d, boxes)
    t_gnd = np.where(d[:, 2] < -1e-6, -cam[2] / np.minimum(d[:, 2], -1e-6), np.inf)
    t = np.minimum(t_box, t_gnd)
    P = cam + d * np.where(np.isfinite(t), t, 0)[:, None]
    sky_c = np.array([0.80, 0.88, 0.95])
    img = np.tile(sky_c, (H * W, 1))
    gnd = np.isfinite(t_gnd) & (t_gnd <= t_box)
    chk = ((np.floor(P[:, 0]) + np.floor(P[:, 1])) % 2).astype(bool)
    img[gnd] = np.where(chk[gnd, None], (0.80, 0.80, 0.76), (0.72, 0.72, 0.68))
    box = np.isfinite(t_box) & (t_box < t_gnd)
    if box.any():
        b = boxes[k[box]]
        rel = P[box, :2] - b[:, :2]
        c, s = np.cos(b[:, 4]), np.sin(b[:, 4])
        lx, ly = c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]
        on_x = np.abs(np.abs(lx) - b[:, 2]) < np.abs(np.abs(ly) - b[:, 3])
        nl = np.where(on_x[:, None], np.stack([np.sign(lx), 0 * lx], -1), np.stack([0 * ly, np.sign(ly)], -1))
        n2 = np.stack([c * nl[:, 0] - s * nl[:, 1], s * nl[:, 0] + c * nl[:, 1]], -1)
        light = np.array([0.5, 0.7, 0.5]) / np.linalg.norm([0.5, 0.7, 0.5])
        lam = np.where(is_top[box], 0.45 + 0.55 * light[2], 0.45 + 0.55 * np.abs(n2 @ light[:2]))
        base = np.where(is_top[box, None], (0.93, 0.78, 0.62), (0.80, 0.55, 0.35))
        if highlight is not None:
            hl = (k[box] == highlight)[:, None]
            base = np.where(hl, np.where(is_top[box, None], (0.55, 0.70, 0.95), (0.20, 0.42, 0.85)), base)
        img[box] = base * lam[:, None] * np.where(is_top[box], 1.0, 0.75 + 0.25 * P[box, 2] / sg.OBS_H)[:, None]
    fog = np.exp(-np.where(np.isfinite(t), t, 60) / 30.0)[:, None]
    img = img * fog + sky_c * (1 - fog)

    def project(Pw):
        rel = (np.asarray(Pw) - cam) @ R
        if rel[0] <= 0.05:
            return None
        return W / 2 - fx * rel[1] / rel[0], H / 2 - fx * rel[2] / rel[0]
    return img.reshape(H, W, 3), project


def ego_truth(occ, seen, vis):
    img = np.zeros((64, 64, 3))
    img[seen & ~occ] = 0.45
    img[vis & ~occ] = (0.15, 0.3, 0.55)
    img[occ & ~seen] = (0.3, 0.08, 0.08)
    img[occ & seen & ~vis] = (1, 0.6, 0)
    img[occ & vis] = (0.9, 0.1, 0.1)
    return img


def prism_edges(b):
    """The 12 edges of box b as a prism of height OBS_H: list of (P0, P1) 3-D points, e.g. for ghost outlines."""
    c = box_corners(b)
    lo, hi = [np.array([*p, 0.0]) for p in c], [np.array([*p, sg.OBS_H]) for p in c]
    return [(lo[i], lo[(i + 1) % 4]) for i in range(4)] + [(hi[i], hi[(i + 1) % 4]) for i in range(4)] + \
        [(lo[i], hi[i]) for i in range(4)]


def even(frame):
    """Crop an (H, W, 3) frame to even sides (H.264 / yuv420p needs them)."""
    return frame[:frame.shape[0] // 2 * 2, :frame.shape[1] // 2 * 2]
