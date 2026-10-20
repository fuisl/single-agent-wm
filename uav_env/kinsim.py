"""Kinematic stand-in for the Isaac Sim drone, for machines where Isaac's RTX renderer does not run.

Same scenes, observations, reward and termination as rl_server.py / collect.py, but:
  * depth is rendered analytically from the box geometry (scenegen.analytic_depth; collect.py --validate
    found it identical to Isaac's render, median error 0 m),
  * dynamics are first-order velocity / yaw tracking (as in dry_run.py) instead of the Hummingbird
    + Lee controller, so attitude is yaw-only and there is no tilt.
Pure numpy; used by kin_collect.py and kin_server.py.
"""
import numpy as np

import scenegen as sg

PHYS_DT, SUBSTEPS = 0.005, 10        # 200 Hz physics, 20 Hz control
TAU_V, TAU_YAW, MAX_YAW_RATE = 0.25, 0.3, 2.5
CAM_OFFSET = np.array([0.15, 0.0, 0.0])


class KinDrone:

    def __init__(self, start_xy, yaw, z=1.5):
        self.pos = np.array([start_xy[0], start_xy[1], z], np.float64)
        self.vel = np.zeros(3)
        self.yaw, self.yaw_rate = float(yaw), 0.0

    def step(self, action):
        """Hold the 5-D action (yaw-frame velocity / LAMBDA, cos / sin of the absolute yaw command) for one
        control step; the body-frame velocity command is re-rotated every physics step, as in rl_server.py."""
        a = np.asarray(action, np.float64)
        yaw_cmd, vb = np.arctan2(a[4], a[3]), a[:3] * sg.LAMBDA
        for _ in range(SUBSTEPS):
            c, s = np.cos(self.yaw), np.sin(self.yaw)
            vw = np.array([c * vb[0] - s * vb[1], s * vb[0] + c * vb[1], vb[2]])
            self.vel += (vw - self.vel) * PHYS_DT / TAU_V
            self.pos = self.pos + self.vel * PHYS_DT
            self.yaw_rate = float(np.clip(sg.wrap(yaw_cmd - self.yaw) / TAU_YAW, -MAX_YAW_RATE, MAX_YAW_RATE))
            self.yaw = sg.wrap(self.yaw + self.yaw_rate * PHYS_DT)

    @property
    def quat(self):
        return np.array([np.cos(self.yaw / 2), 0.0, 0.0, np.sin(self.yaw / 2)])

    def depth(self, boxes):
        """(48, 80) distance to image plane in [D_MIN, D_MAX] m."""
        R = sg.quat_to_rot(self.quat)
        d = sg.analytic_depth(self.pos + R @ CAM_OFFSET, R, boxes)
        return np.clip(np.nan_to_num(d, nan=sg.D_MAX, posinf=sg.D_MAX), sg.D_MIN, sg.D_MAX)

    def state(self, goal):
        return sg.obs15(self.pos, self.quat, self.vel, np.array([0.0, 0.0, self.yaw_rate]), goal)
