"""Isaac Sim environment server for online RL (runs in the isaac41 env).

N drones live in separate arenas of one stage and are stepped together. Each DreamerV3 env process
(isaac_uav_env.py, dreamer env) connects as one slot and, every control step, sends either
("step", action) or ("reset", options). When all N slots have sent a request, the server applies the
actions for 10 physics steps (200 Hz -> 20 Hz control), renders once and answers every slot with
its new observation. Observations, actions, reward and termination follow the AirDreamer paper
(48x80 depth in [0.6, 6] m, 15-D state, 5-D action, 20 Hz control / 200 Hz physics).

    python rl_server.py --num_envs 16 --port 6100
"""
import argparse
import os
import time
from multiprocessing.connection import Listener, wait

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--port", type=int, default=6100)
parser.add_argument("--authkey", default="airdreamer")
parser.add_argument("--extras", action="store_true", help="also send pos / yaw (evaluation only, not in obs_space)")
parser.add_argument("--max_steps", type=int, default=1200)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--trap_start_frac", type=float, default=0.0,
                    help="training: fraction of random episodes that start inside a U or C trap")
args = parser.parse_args()

from omegaconf import OmegaConf
from omni_drones import init_simulation_app

OD_ROOT = os.path.dirname(os.path.dirname(__import__("omni_drones").__file__))
cfg = OmegaConf.load(os.path.join(OD_ROOT, "examples", "demo.yaml"))
cfg.headless = True
PHYS_DT, SUBSTEPS = 0.005, 10
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
pools = [[VisualCuboid(prim_path=f"/World/obstacles/env_{i}/b_{j}", position=HIDE, size=1.0,
                       color=np.array([0.6, 0.62, 0.66])) for j in range(sg.MAX_BOXES)] for i in range(N)]
H_AP = 20.955
F_D455 = H_AP / (2 * np.tan(sg.HFOV / 2))
CAM_OFFSET = np.array([0.15, 0.0, 0.0])
cam = Camera(PinholeCameraCfg(sensor_tick=0, resolution=(80, 48), data_types=["distance_to_image_plane"],
                              usd_params=PinholeCameraCfg.UsdCameraCfg(focal_length=F_D455, horizontal_aperture=H_AP,
                                                                       clipping_range=(0.3, 1000.0))))
cam_paths = [f"/World/envs/env_0/{drone.name}_{i}/base_link/DepthCam" for i in range(N)]
cam.spawn(cam_paths, translations=[tuple(CAM_OFFSET)] * N, targets=[tuple(CAM_OFFSET + [1, 0, 0])] * N)
for p in cam_paths:
    prim_utils.get_prim_at_path(p).GetAttribute("verticalAperture").Set(H_AP * 48 / 80)
sim.reset()
drone.initialize()
cam.initialize(f"/World/envs/env_0/{drone.name}_.*/base_link/DepthCam")
found = prim_utils.find_matching_prim_paths(f"/World/envs/env_0/{drone.name}_.*/base_link/DepthCam")
cam_order = [found.index(p) for p in cam_paths]
drone_paths = list(drone._view.prim_paths)
drone_order = [drone_paths.index(f"/World/envs/env_0/{drone.name}_{i}") for i in range(N)]
LAMBDA_T = torch.tensor(sg.LAMBDA, device=device, dtype=torch.float32)


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class Slot:
    def __init__(self, i):
        self.i = i
        self.rng = np.random.default_rng(args.seed * 1000 + i)
        self.scene = None
        self.done = True

    def new_episode(self, options):
        rng = np.random.default_rng(options["seed"]) if options.get("seed") is not None else self.rng
        kind = options.get("kind", "random")
        if kind == "random" and rng.random() < args.trap_start_frac:
            self.scene = sg.sample_test_scene(rng, rng.choice(["utrap_in", "ctrap"]), jitter=True)
        elif kind == "random":
            self.scene = sg.sample_scene(rng)
        else:
            self.scene = sg.sample_test_scene(rng, kind)
        self.d_prev = float(np.linalg.norm(self.scene.goal - self.scene.start))
        self.t, self.done = 0, False
        boxes = self.scene.boxes
        for j, prim in enumerate(pools[self.i]):
            if j < len(boxes):
                cx, cy, hx, hy, yaw = boxes[j]
                prim.set_world_pose(np.array([cx + ORIGINS[self.i, 0], cy + ORIGINS[self.i, 1], sg.OBS_H / 2]),
                                    yaw_quat(yaw))
                prim.set_local_scale(np.array([2 * hx, 2 * hy, sg.OBS_H]))
            else:
                prim.set_world_pose(HIDE, yaw_quat(0.0))
        yaw0 = self.scene.start_yaw if getattr(self.scene, "start_yaw", None) is not None else \
            np.arctan2(*(self.scene.goal - self.scene.start)[::-1]) + rng.uniform(-0.6, 0.6)
        self.yaw0 = yaw0
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


def physics(actions):
    """Hold the (N, 5) actions for SUBSTEPS physics steps; body-frame velocity re-rotated every step."""
    a_view = torch.zeros((N, 5), device=device)
    a_view[drone_order] = torch.as_tensor(actions, device=device, dtype=torch.float32)
    yaw_cmd = torch.atan2(a_view[:, 4], a_view[:, 3])
    vb = a_view[:, :3] * LAMBDA_T
    for _ in range(SUBSTEPS):
        st = drone.get_state()[..., :13]
        q = st[0, :, 3:7]
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        c, s = torch.cos(yaw), torch.sin(yaw)
        vw = torch.stack([c * vb[:, 0] - s * vb[:, 1], s * vb[:, 0] + c * vb[:, 1], vb[:, 2]], -1)
        drone.apply_action(controller.compute(st, target_vel=vw[None], target_yaw=yaw_cmd[None, :, None]))
        sim.step(render=False)
    sim.render()


def read_obs(slots):
    depth_all = cam.get_images()["distance_to_image_plane"][:, 0].cpu().numpy()
    st = drone.get_state()[..., :13].squeeze(0)[drone_order].cpu().numpy().astype(np.float64)
    vel = drone.get_velocities().squeeze(0)[drone_order].cpu().numpy().astype(np.float64)
    out = []
    for i, slot in enumerate(slots):
        pos = st[i, :3].copy(); pos[:2] -= ORIGINS[i]
        q = st[i, 3:7]
        depth = np.clip(np.nan_to_num(depth_all[cam_order[i]], nan=sg.D_MAX, posinf=sg.D_MAX), sg.D_MIN, sg.D_MAX)
        first = slot.t == 0
        r, slot.d_prev, status, _ = sg.reward_terms(pos, slot.scene.goal, slot.d_prev, slot.scene.boxes)
        if first:
            r, status = 0.0, None
        timeout = slot.t + 1 >= args.max_steps or np.abs(pos[:2]).max() > sg.EXTENT
        last = status is not None or timeout
        obs = {
            "depth": np.round((depth - sg.D_MIN) / (sg.D_MAX - sg.D_MIN) * 255).astype(np.uint8)[..., None],
            "state": sg.obs15(pos, q, vel[i, :3], vel[i, 3:], slot.scene.goal),
            "reward": np.float32(r), "is_first": first, "is_last": last, "is_terminal": status is not None,
            "log/goal": np.float32(status == "goal"), "log/collision": np.float32(status == "collision"),
            "log/height": np.float32(status == "height"), "log/timeout": np.float32(last and status is None),
        }
        if args.extras:
            obs["pos"], obs["yaw"] = pos.astype(np.float32), np.float32(sg.yaw_of(q))
        slot.t += 1
        slot.done = last
        out.append(obs)
    return out


# ---------------------------------------------------------------- serve
slots = [Slot(i) for i in range(N)]
listener = Listener(("127.0.0.1", args.port), authkey=args.authkey.encode(), backlog=128)
print(f"SERVER_READY port={args.port} envs={N}", flush=True)
conns = {}
while len(conns) < N:                                   # every client introduces itself with its slot index
    c = listener.accept()
    msg = c.recv()
    assert msg[0] == "hello", msg
    conns[int(msg[1])] = c
    print("client", msg[1], "connected", flush=True)

pending = {}
steps, t0 = 0, time.time()
tsum = {"wait": 0.0, "reset": 0.0, "physics": 0.0, "obs": 0.0, "send": 0.0}
hover = np.array([0, 0, 0, 1, 0], np.float32)
while True:
    ta = time.time()
    for c in wait(list(conns.values())):
        i = next(k for k, v in conns.items() if v is c)
        try:
            pending[i] = c.recv()
        except EOFError:
            print("client", i, "disconnected, shutting down", flush=True)
            simulation_app.close(); raise SystemExit
    if len(pending) < N:
        continue
    tb = time.time(); tsum["wait"] += tb - ta
    actions = np.tile(hover, (N, 1))
    resets = []
    for i, msg in pending.items():
        if msg[0] == "reset" or slots[i].done:
            resets.append(i)
        else:
            actions[i] = msg[1]
    if resets:
        teleport(resets, [slots[i].new_episode(pending[i][1] if pending[i][0] == "reset" else {}) for i in resets])
        for i in resets:                                # hover right after a teleport
            actions[i] = [0, 0, 0, np.cos(slots[i].yaw0), np.sin(slots[i].yaw0)]
    tc = time.time(); tsum["reset"] += tc - tb
    if resets:                                          # moved obstacles show up in the next render; flush it
        sim.render()
    physics(actions)
    td = time.time(); tsum["physics"] += td - tc
    obs = read_obs(slots)
    te = time.time(); tsum["obs"] += te - td
    for i in range(N):
        conns[i].send(obs[i])
    tsum["send"] += time.time() - te
    pending.clear()
    steps += 1
    if steps % 500 == 0:
        ms = {k: round(v / steps * 1000, 1) for k, v in tsum.items()}
        print(f"server steps={steps} ({steps * N / (time.time() - t0):.0f} agent-steps/s) ms/round={ms}", flush=True)
