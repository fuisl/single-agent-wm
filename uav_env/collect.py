"""Collect world-model training data in Isaac Sim 4.1 + OmniDrones (headless, many drones in parallel).

Each drone flies in its own 24 x 24 m arena (arenas are 40 m apart, beyond the 6 m depth range).
Obstacles are visual-only boxes re-placed every episode; collisions are detected analytically
(distance < 0.3 m, as in the sandbox), so the drone physics never needs to touch them.

Per 20 Hz step and drone we store (DreamerV3 convention: action = a_t chosen after seeing obs_t):
  depth (48, 80) float16 in [0.6, 6] m, state (15,), action (5,), reward, is_first/is_last/is_terminal,
  occ / seen / vis (64, 64) bool ego-centric labels, pos (3,), yaw.

    python collect.py --num_envs 16 --episodes 2000 --out data/wm_v1
    python collect.py --num_envs 4 --episodes 8 --out data/wm_test --validate
"""
import argparse
import json
import os
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=2000)
parser.add_argument("--out", default="data/wm_v1")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_steps", type=int, default=1200)          # 60 s at 20 Hz
parser.add_argument("--validate", action="store_true", help="compare rendered depth with analytic depth")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

from omegaconf import OmegaConf
from omni_drones import init_simulation_app

OD_ROOT = os.path.dirname(os.path.dirname(__import__("omni_drones").__file__))
cfg = OmegaConf.load(os.path.join(OD_ROOT, "examples", "demo.yaml"))
cfg.headless = True
PHYS_DT, SUBSTEPS = 0.005, 10                                       # 200 Hz physics, 20 Hz control
cfg.sim.dt = PHYS_DT
simulation_app = init_simulation_app(cfg)

import torch
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.simulation_context import SimulationContext
import omni_drones.utils.scene as scene_utils
from omni_drones.robots.drone import MultirotorBase
from omni_drones.sensors.camera import Camera, PinholeCameraCfg

import scenegen as sg

sim = SimulationContext(stage_units_in_meters=1.0, physics_dt=PHYS_DT, rendering_dt=PHYS_DT * SUBSTEPS,
                        sim_params=cfg.sim, backend="torch", device=cfg.sim.device)
device = sim.device
N = args.num_envs
cols = int(np.ceil(np.sqrt(N)))
ORIGINS = np.array([[40.0 * (i % cols), 40.0 * (i // cols)] for i in range(N)])
HIDE = np.array([0.0, 0.0, -50.0])

# ---------------------------------------------------------------- stage
drone, controller = MultirotorBase.make(cfg.drone_model.name, cfg.drone_model.controller, device)
drone.spawn(translations=[(*ORIGINS[i], 1.5) for i in range(N)])
scene_utils.design_scene()
prim_utils.create_prim("/World/Light/Dome", "DomeLight", attributes={"inputs:intensity": 600.0})

pools = []
for i in range(N):
    pool = [VisualCuboid(prim_path=f"/World/obstacles/env_{i}/b_{j}", position=HIDE, size=1.0,
                         color=np.array([0.6, 0.62, 0.66])) for j in range(sg.MAX_BOXES)]
    pools.append(pool)

H_AP = 20.955
F_D455 = H_AP / (2 * np.tan(sg.HFOV / 2))
CAM_OFFSET = np.array([0.15, 0.0, 0.0])
cam_cfg = PinholeCameraCfg(sensor_tick=0, resolution=(80, 48), data_types=["distance_to_image_plane"],
                           usd_params=PinholeCameraCfg.UsdCameraCfg(focal_length=F_D455, horizontal_aperture=H_AP,
                                                                    clipping_range=(0.3, 1000.0)))
cam = Camera(cam_cfg)
cam_paths = [f"/World/envs/env_0/{drone.name}_{i}/base_link/DepthCam" for i in range(N)]
cam.spawn(cam_paths, translations=[tuple(CAM_OFFSET)] * N, targets=[tuple(CAM_OFFSET + [1, 0, 0])] * N)
for p in cam_paths:
    prim_utils.get_prim_at_path(p).GetAttribute("verticalAperture").Set(H_AP * 48 / 80)

sim.reset()
drone.initialize()
cam.initialize(f"/World/envs/env_0/{drone.name}_.*/base_link/DepthCam")
found = prim_utils.find_matching_prim_paths(f"/World/envs/env_0/{drone.name}_.*/base_link/DepthCam")
cam_order = [found.index(p) for p in cam_paths]                     # annotator index for drone i
drone_paths = list(drone._view.prim_paths)
drone_order = [drone_paths.index(f"/World/envs/env_0/{drone.name}_{i}") for i in range(N)]
print("camera order", cam_order[:8], "drone order", drone_order[:8], flush=True)


# ---------------------------------------------------------------- helpers
def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class EnvSlot:
    def __init__(self, i, rng):
        self.i, self.rng = i, rng
        self.buf = None

    def new_episode(self):
        self.mode = self.rng.choice(["goal", "explore", "trap"], p=[0.7, 0.15, 0.15])
        self.scene = sg.sample_scene(self.rng, need_trap=(self.mode == "trap"))
        self.pilot = sg.Behaviour(self.scene, self.mode, self.rng)
        self.seen = sg.SeenMap()
        self.d_prev = float(np.linalg.norm(self.scene.goal - self.scene.start))
        self.t = 0
        self.buf = {k: [] for k in ["depth", "state", "action", "reward", "is_first", "is_last", "is_terminal",
                                    "occ", "seen", "vis", "pos", "yaw"]}
        boxes = self.scene.boxes
        for j, prim in enumerate(pools[self.i]):
            if j < len(boxes):
                cx, cy, hx, hy, yaw = boxes[j]
                prim.set_world_pose(np.array([cx + ORIGINS[self.i, 0], cy + ORIGINS[self.i, 1], sg.OBS_H / 2]),
                                    yaw_quat(yaw))
                prim.set_local_scale(np.array([2 * hx, 2 * hy, sg.OBS_H]))
            else:
                prim.set_world_pose(HIDE, yaw_quat(0.0))
        yaw0 = np.arctan2(*(self.scene.goal - self.scene.start)[::-1]) + self.rng.uniform(-0.6, 0.6)
        return np.array([*(self.scene.start + ORIGINS[self.i]), 1.5]), yaw_quat(yaw0)


def teleport(idx, poses):
    pos, rot = drone.get_world_poses()
    pos, rot = pos.clone(), rot.clone()
    vel = drone.get_velocities().clone()
    for i, (p, q) in zip(idx, poses):
        k = drone_order[i]
        pos[0, k] = torch.tensor(p, device=device, dtype=pos.dtype)
        rot[0, k] = torch.tensor(q, device=device, dtype=rot.dtype)
        vel[0, k] = 0
    drone.set_world_poses(pos, rot)
    drone.set_velocities(vel)


def read_state():
    st = drone.get_state()[..., :13].squeeze(0)                      # (N, 13) in drone view order
    vel = drone.get_velocities().squeeze(0)                          # (N, 6) world lin + ang
    st = st[drone_order].cpu().numpy().astype(np.float64)
    vel = vel[drone_order].cpu().numpy().astype(np.float64)
    return st[:, :3], st[:, 3:7], vel[:, :3], vel[:, 3:]


# ---------------------------------------------------------------- main loop
rngs = [np.random.default_rng(args.seed * 1000 + i) for i in range(N)]
slots = [EnvSlot(i, rngs[i]) for i in range(N)]
teleport(range(N), [s.new_episode() for s in slots])
actions = np.zeros((N, 5), np.float32)
actions[:, 3] = 1.0
episodes_done, steps_done, t_start = 0, 0, time.time()
outcomes, val_err = {}, []
summary_path = os.path.join(args.out, "progress.json")

while episodes_done < args.episodes:
    # ---- physics: hold each drone's action for SUBSTEPS physics steps
    a_t = torch.tensor(actions, device=device)
    for _ in range(SUBSTEPS):
        st = drone.get_state()[..., :13]                             # (1, N, 13) view order
        q = st[0, :, 3:7]
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        a_view = torch.zeros_like(a_t)
        a_view[drone_order] = a_t
        vb = a_view[:, :3] * torch.tensor(sg.LAMBDA, device=device, dtype=a_t.dtype)
        c, s = torch.cos(yaw), torch.sin(yaw)
        vw = torch.stack([c * vb[:, 0] - s * vb[:, 1], s * vb[:, 0] + c * vb[:, 1], vb[:, 2]], -1)
        yaw_cmd = torch.atan2(a_view[:, 4], a_view[:, 3])
        cmd = controller.compute(st, target_vel=vw[None], target_yaw=yaw_cmd[None, :, None])
        drone.apply_action(cmd)
        sim.step(render=False)
    sim.render()

    depth_all = cam.get_images()["distance_to_image_plane"][:, 0].cpu().numpy()   # (N, 48, 80) annotator order
    pos_w, quat, vel_w, ang_w = read_state()
    resets = []
    for i, slot in enumerate(slots):
        pos = pos_w[i].copy(); pos[:2] -= ORIGINS[i]
        q = quat[i]; yaw = sg.yaw_of(q)
        d_raw = depth_all[cam_order[i]]
        depth = np.clip(np.nan_to_num(d_raw, nan=sg.D_MAX, posinf=sg.D_MAX), sg.D_MIN, sg.D_MAX)
        if args.validate:
            R = sg.quat_to_rot(q)
            ana = sg.analytic_depth(pos + R @ CAM_OFFSET, R, slot.scene.boxes)
            ana = np.clip(np.nan_to_num(ana, posinf=sg.D_MAX), sg.D_MIN, sg.D_MAX)
            val_err.append(float(np.median(np.abs(ana - depth))))
        _, occ, seen, vis = sg.labels(pos, yaw, slot.scene.boxes, slot.seen)
        first = slot.t == 0
        r, slot.d_prev, status, _ = sg.reward_terms(pos, slot.scene.goal, slot.d_prev, slot.scene.boxes)
        if first:
            r, status = 0.0, None
        out_of_arena = np.abs(pos[:2]).max() > sg.EXTENT
        last = status is not None or slot.t + 1 >= args.max_steps or out_of_arena
        terminal = status is not None
        act = slot.pilot.act(pos, yaw, slot.t * 0.05) if not last else np.zeros(5, np.float32)
        b = slot.buf
        b["depth"].append(depth.astype(np.float16)); b["state"].append(sg.obs15(pos, q, vel_w[i], ang_w[i], slot.scene.goal))
        b["action"].append(act); b["reward"].append(np.float32(r))
        b["is_first"].append(first); b["is_last"].append(last); b["is_terminal"].append(terminal)
        b["occ"].append(occ); b["seen"].append(seen); b["vis"].append(vis)
        b["pos"].append(pos.astype(np.float32)); b["yaw"].append(np.float32(yaw))
        actions[i] = act
        slot.t += 1
        steps_done += 1
        if last:
            outcome = status or ("timeout" if slot.t >= args.max_steps else "out_of_arena")
            outcomes.setdefault(slot.mode, {}).setdefault(outcome, 0)
            outcomes[slot.mode][outcome] += 1
            fname = os.path.join(args.out, f"ep_{episodes_done:05d}_env{i:02d}_{slot.mode}_{outcome}.npz")
            np.savez_compressed(fname, **{k: np.stack(v) for k, v in b.items()},
                                boxes=slot.scene.boxes.astype(np.float32), start=slot.scene.start,
                                goal=slot.scene.goal, mode=slot.mode, outcome=outcome)
            episodes_done += 1
            resets.append(i)
    if resets:
        teleport(resets, [slots[i].new_episode() for i in resets])
        for i in resets:
            actions[i] = 0; actions[i, 3] = 1.0
    if resets and episodes_done % 20 < len(resets):
        el = time.time() - t_start
        info = {"episodes": episodes_done, "steps": steps_done, "steps_per_s": round(steps_done / el, 1),
                "elapsed_min": round(el / 60, 1), "outcomes": outcomes}
        if val_err:
            info["depth_median_abs_err_m"] = round(float(np.median(val_err)), 4)
            info["depth_err_p95_m"] = round(float(np.percentile(val_err, 95)), 4)
        json.dump(info, open(summary_path, "w"), indent=1)
        print(json.dumps(info), flush=True)

el = time.time() - t_start
info = {"episodes": episodes_done, "steps": steps_done, "steps_per_s": round(steps_done / el, 1),
        "elapsed_min": round(el / 60, 1), "outcomes": outcomes}
if val_err:
    info["depth_median_abs_err_m"] = round(float(np.median(val_err)), 4)
    info["depth_err_p95_m"] = round(float(np.percentile(val_err, 95)), 4)
json.dump(info, open(summary_path, "w"), indent=1)
print("COLLECT_DONE", json.dumps(info), flush=True)
simulation_app.close()
