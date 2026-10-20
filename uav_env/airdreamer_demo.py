"""AirDreamer-style depth-navigation demo scene in OmniDrones (Isaac Sim 4.1, headless).

A Hummingbird quadrotor with a forward depth camera (48x80, 87 deg HFOV, valid range [0.6, 6] m,
as in the AirDreamer paper) flies a pre-planned path through pillars, around a large wall and
around a U-shaped trap. The flight uses OmniDrones' LeePositionController tracking a spline; it is
NOT a learned policy. Output: a composited 1920x1080 mp4 plus raw 48x80 depth frames (.npy).
"""
import argparse
import os
import subprocess

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="outputs/demo")
parser.add_argument("--speed", type=float, default=1.5)
parser.add_argument("--max_seconds", type=float, default=60.0)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

from omegaconf import OmegaConf
from omni_drones import init_simulation_app

OD_ROOT = os.path.dirname(os.path.dirname(__import__("omni_drones").__file__))
cfg = OmegaConf.load(os.path.join(OD_ROOT, "examples", "demo.yaml"))
cfg.headless = True
PHYS_DT = 0.005          # 200 Hz physics, as in the paper
REC_EVERY = 6            # record at ~33 Hz
cfg.sim.dt = PHYS_DT
simulation_app = init_simulation_app(cfg)

import cv2
import torch
import dataclasses
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.objects import FixedCuboid, VisualSphere
from omni.isaac.core.prims import XFormPrim
import omni_drones.utils.scene as scene_utils
from omni_drones.robots.drone import MultirotorBase
from omni_drones.sensors.camera import Camera, PinholeCameraCfg, orientation_from_view

sim = SimulationContext(
    stage_units_in_meters=1.0,
    physics_dt=PHYS_DT,
    rendering_dt=PHYS_DT * REC_EVERY,
    sim_params=cfg.sim,
    backend="torch",
    device=cfg.sim.device,
)
device = sim.device

# ---------------------------------------------------------------- scene
ALT = 1.5
START = (0.0, 0.0)
GOAL = (30.0, 0.0)

obstacles = []  # (name, center xyz, size xyz, color)
def box(name, x, y, sx, sy, h, color):
    obstacles.append((name, (x, y, h / 2), (sx, sy, h), color))

PILLAR = (0.45, 0.55, 0.70)
for i, (x, y) in enumerate([(5.0, 1.0), (5.0, -1.3), (6.8, 0.0), (8.5, -1.3), (8.5, 2.3), (10.0, 0.9)]):
    box(f"pillar_{i}", x, y, 0.6, 0.6, 4.0, PILLAR)
WALL = (0.80, 0.80, 0.82)
box("big_wall", 13.0, -1.5, 0.4, 9.0, 4.0, WALL)           # spans y in [-6, 3]
U = (0.90, 0.55, 0.20)
box("u_left", 20.0, 2.2, 4.0, 0.3, 3.0, U)                  # U-trap opening toward -x
box("u_right", 20.0, -2.2, 4.0, 0.3, 3.0, U)
box("u_back", 22.0, 0.0, 0.3, 4.7, 3.0, U)
CLUTTER = (0.55, 0.45, 0.40)
for i, (x, y, s, h) in enumerate([(3.0, -3.5, 1.0, 1.2), (7.0, -3.8, 1.2, 2.0), (9.0, 3.8, 1.0, 2.5),
                                  (16.0, -3.0, 1.5, 1.5), (24.0, -3.5, 1.2, 2.2), (28.0, -3.0, 1.0, 1.0),
                                  (25.5, 4.6, 1.0, 1.8), (27.5, 2.2, 0.8, 1.2)]):
    box(f"clutter_{i}", x, y, s, s, h, CLUTTER)

for name, c, s, color in obstacles:
    FixedCuboid(prim_path=f"/World/obstacles/{name}", position=np.array(c), scale=np.array(s),
                color=np.array(color))
VisualSphere(prim_path="/World/goal", position=np.array([GOAL[0], GOAL[1], ALT]), radius=0.3,
             color=np.array([0.1, 0.9, 0.2]))

drone, controller = MultirotorBase.make(cfg.drone_model.name, cfg.drone_model.controller, device)
drone.spawn(translations=[(START[0], START[1], ALT)])
scene_utils.design_scene()
prim_utils.create_prim("/World/Light/Dome", "DomeLight", attributes={"inputs:intensity": 600.0})


def set_vertical_aperture(path, h_ap, res):
    prim_utils.get_prim_at_path(path).GetAttribute("verticalAperture").Set(h_ap * res[1] / res[0])


# Onboard depth camera: 80x48, HFOV 87 deg -> focal = ap / (2 tan(43.5 deg))
H_AP = 20.955
F_D455 = H_AP / (2 * np.tan(np.deg2rad(87 / 2)))
mount = f"/World/envs/env_0/{drone.name}_0/base_link"
onboard_usd = PinholeCameraCfg.UsdCameraCfg(focal_length=F_D455, horizontal_aperture=H_AP,
                                            clipping_range=(0.3, 1000.0))  # near clip hides own rotors
depth_cfg = PinholeCameraCfg(sensor_tick=0, resolution=(80, 48),
                             data_types=["distance_to_image_plane"], usd_params=onboard_usd)
hires_cfg = PinholeCameraCfg(sensor_tick=0, resolution=(640, 384),
                             data_types=["rgb", "distance_to_image_plane"], usd_params=onboard_usd)
depth_cam, hires_cam = Camera(depth_cfg), Camera(hires_cfg)
depth_cam.spawn([f"{mount}/DepthCam"], translations=[(0.15, 0, 0)], targets=[(1.15, 0, 0)])
hires_cam.spawn([f"{mount}/HiresCam"], translations=[(0.15, 0, 0)], targets=[(1.15, 0, 0)])
set_vertical_aperture(f"{mount}/DepthCam", H_AP, depth_cfg.resolution)
set_vertical_aperture(f"{mount}/HiresCam", H_AP, hires_cfg.resolution)

chase_cfg = PinholeCameraCfg(sensor_tick=0, resolution=(1280, 720), data_types=["rgb"],
                             usd_params=PinholeCameraCfg.UsdCameraCfg(focal_length=16.0, horizontal_aperture=H_AP,
                                                                      clipping_range=(0.1, 1000.0)))
over_cfg = dataclasses.replace(chase_cfg, resolution=(1280, 360))
chase_cam, over_cam = Camera(chase_cfg), Camera(over_cfg)
OVER_EYE, OVER_AT = (15.0, -21.0, 16.0), (15.0, 0.5, 0.0)
chase_cam.spawn(["/World/ChaseCam"], translations=[(-4, 0, 3)], targets=[(0, 0, 1.5)])
over_cam.spawn(["/World/OverviewCam"], translations=[OVER_EYE], targets=[OVER_AT])
set_vertical_aperture("/World/ChaseCam", H_AP, chase_cfg.resolution)
set_vertical_aperture("/World/OverviewCam", H_AP, over_cfg.resolution)

sim.reset()
drone.initialize()
depth_cam.initialize(f"{mount}/DepthCam")
hires_cam.initialize(f"{mount}/HiresCam")
chase_cam.initialize("/World/ChaseCam")
over_cam.initialize("/World/OverviewCam")
chase_xf = XFormPrim("/World/ChaseCam")

# ---------------------------------------------------------------- reference path
WAYPOINTS = np.array([
    START, (2.5, 0.0), (5.0, -0.15), (6.8, 1.3), (8.5, 0.35), (10.0, -0.7), (11.6, 1.8),
    (13.0, 4.6), (14.8, 3.6), (17.0, 3.9), (20.0, 4.0), (23.2, 3.4), (25.0, 1.2), (27.0, 0.1), GOAL,
])


def catmull_rom(pts, n_per=60):
    p = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, n_per, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1])
    return np.array(out)


dense = catmull_rom(WAYPOINTS)
seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
arclen = np.concatenate([[0], np.cumsum(seg)])
PATH_LEN = arclen[-1]


def ref_at(s):
    s = np.clip(s, 0, PATH_LEN)
    x = np.interp(s, arclen, dense[:, 0]); y = np.interp(s, arclen, dense[:, 1])
    s2 = min(s + 0.3, PATH_LEN); s1 = max(s2 - 0.6, 0)
    dx = np.interp(s2, arclen, dense[:, 0]) - np.interp(s1, arclen, dense[:, 0])
    dy = np.interp(s2, arclen, dense[:, 1]) - np.interp(s1, arclen, dense[:, 1])
    return np.array([x, y]), np.arctan2(dy, dx), np.array([dx, dy]) / (np.hypot(dx, dy) + 1e-6)


# ---------------------------------------------------------------- video writer
W, H = 1920, 1080
ffmpeg = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
     "-r", f"{1 / (PHYS_DT * REC_EVERY):.3f}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
     "-crf", "20", os.path.join(args.out, "airdreamer_demo.mp4")],
    stdin=subprocess.PIPE)

D_MIN, D_MAX = 0.6, 6.0


def depth_to_color(d):
    d = np.nan_to_num(d, nan=D_MAX, posinf=D_MAX, neginf=D_MAX)
    d = np.clip(d, D_MIN, D_MAX)
    g = ((D_MAX - d) / (D_MAX - D_MIN) * 255).astype(np.uint8)  # near = hot
    return cv2.cvtColor(cv2.applyColorMap(g, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def label(img, text, org=(12, 30), scale=0.8):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def fit(img, w, h):
    canvas = np.zeros((h, w, 3), np.uint8)
    ih, iw = img.shape[:2]
    k = min(w / iw, h / ih)
    r = cv2.resize(img, (int(iw * k), int(ih * k)), interpolation=cv2.INTER_NEAREST)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return canvas


def _over_basis():
    eye, at = np.array(OVER_EYE), np.array(OVER_AT)
    fwd = (at - eye) / np.linalg.norm(at - eye)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    return eye, fwd, right, np.cross(right, fwd)


_OB = _over_basis()
_F_OVER = 16.0 / H_AP * 1280


def project_overview(p):
    eye, fwd, right, up = _OB
    v = np.asarray(p) - eye
    z = v @ fwd
    return int(640 + _F_OVER * (v @ right) / z), int(180 - _F_OVER * (v @ up) / z)


def draw_overview(img, traj):
    pts = [project_overview(q[1:4]) for q in traj[::2]]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, a, b, (230, 30, 30), 2, cv2.LINE_AA)
    if pts:
        cv2.circle(img, pts[-1], 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, pts[-1], 5, (230, 30, 30), -1, cv2.LINE_AA)


def to_np(t):
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------- main loop
HOVER_S = 1.5
s_progress = 0.0
cam_heading = np.array([1.0, 0.0])
raw_depth, traj = [], []
min_clear = np.inf
state = drone.get_state()[..., :13].squeeze(0)
n_steps = int(args.max_seconds / PHYS_DT)
reached_at = None
frame = 0
for step in range(n_steps):
    t = step * PHYS_DT
    if t > HOVER_S:
        s_progress = min(s_progress + args.speed * PHYS_DT, PATH_LEN)
    xy, yaw, tangent = ref_at(s_progress)
    moving = HOVER_S < t and s_progress < PATH_LEN
    tgt_pos = torch.tensor([[xy[0], xy[1], ALT]], device=device, dtype=torch.float32)
    tgt_vel = torch.tensor([[*(tangent * args.speed * moving), 0.0]], device=device, dtype=torch.float32)
    tgt_yaw = torch.tensor([[yaw]], device=device, dtype=torch.float32)
    action = controller.compute(state, target_pos=tgt_pos, target_vel=tgt_vel, target_yaw=tgt_yaw)
    drone.apply_action(action)
    sim.step(render=False)
    state = drone.get_state()[..., :13].squeeze(0)
    pos = to_np(state[0, :3])

    if s_progress >= PATH_LEN and reached_at is None and np.linalg.norm(pos[:2] - np.array(GOAL)) < 0.3:
        reached_at = t

    if step % REC_EVERY:
        continue
    # chase camera: behind and above the drone, heading low-pass filtered
    vel_xy = to_np(state[0, 7:9])
    if np.linalg.norm(vel_xy) > 0.3:
        cam_heading = 0.93 * cam_heading + 0.07 * vel_xy / np.linalg.norm(vel_xy)
        cam_heading /= np.linalg.norm(cam_heading)
    eye = (pos[0] - 5.0 * cam_heading[0], pos[1] - 5.0 * cam_heading[1], pos[2] + 3.0)
    at = (pos[0] + 2.0 * cam_heading[0], pos[1] + 2.0 * cam_heading[1], pos[2] - 0.3)
    chase_xf.set_world_pose(np.array(eye), np.array(orientation_from_view(eye, at)))
    sim.render()

    d_raw = to_np(depth_cam.get_images()["distance_to_image_plane"][0, 0])
    hi = hires_cam.get_images()
    hi_rgb = to_np(hi["rgb"][0].permute(1, 2, 0))[..., :3].copy()
    hi_d = to_np(hi["distance_to_image_plane"][0, 0])
    chase = to_np(chase_cam.get_images()["rgb"][0].permute(1, 2, 0))[..., :3].copy()
    over = to_np(over_cam.get_images()["rgb"][0].permute(1, 2, 0))[..., :3].copy()

    d_obs = np.clip(np.nan_to_num(d_raw, nan=D_MAX, posinf=D_MAX), D_MIN, D_MAX)  # paper: fill out-of-range with 6 m
    raw_depth.append(d_obs.astype(np.float16))
    traj.append([t, *pos, yaw, s_progress])
    min_clear = min(min_clear, float(d_obs.min()))

    out = np.zeros((H, W, 3), np.uint8)
    out[:720, :1280] = label(chase, f"Isaac Sim 4.1 + OmniDrones   t = {t:5.1f} s   progress {s_progress / PATH_LEN * 100:5.1f} %")
    draw_overview(over, traj)
    out[720:, :1280] = label(over, "Overview: pillars -> big wall -> U-shaped trap -> goal (green)")
    out[:360, 1280:] = label(fit(hi_rgb, 640, 360), "Onboard RGB (reference only)", scale=0.7)
    out[360:720, 1280:] = label(fit(depth_to_color(hi_d), 640, 360), "Onboard depth, high-res view", scale=0.7)
    out[720:, 1280:] = label(fit(depth_to_color(d_obs), 640, 360),
                             "Policy input: 48x80 depth, [0.6, 6] m", scale=0.7)
    label(out, "near", (1290, 1065), 0.6); label(out, "far = 6 m (purple)", (1660, 1065), 0.6)
    cv2.rectangle(out, (0, 0), (W - 1, H - 1), (40, 40, 40), 2)
    ffmpeg.stdin.write(out.tobytes())
    if frame in (60, 250, 450, 650):
        cv2.imwrite(os.path.join(args.out, f"frame_{frame:04d}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    frame += 1
    if reached_at is not None and t > reached_at + 1.5:
        break

ffmpeg.stdin.close(); ffmpeg.wait()
np.save(os.path.join(args.out, "depth_48x80.npy"), np.stack(raw_depth))
np.save(os.path.join(args.out, "trajectory.npy"), np.array(traj))
tr = np.array(traj)
clear = []
for q in tr:
    best = np.inf
    for name, c, sz, _ in obstacles:
        d = np.maximum(np.abs(q[1:4] - np.array(c)) - np.array(sz) / 2, 0)
        best = min(best, float(np.linalg.norm(d)))
    clear.append(best)
import json
json.dump({"frames": frame, "reached_at": reached_at, "path_len": float(PATH_LEN),
           "min_obstacle_clearance_m": float(np.min(clear)), "min_depth_px_m": min_clear,
           "z_min": float(tr[:, 3].min()), "z_max": float(tr[:, 3].max())},
          open(os.path.join(args.out, "summary.json"), "w"), indent=1)
print(f"DEMO_DONE frames={frame} reached_at={reached_at} path_len={PATH_LEN:.1f}m min_depth={min_clear:.2f}m")
simulation_app.close()
