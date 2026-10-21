"""Scene sampling, geometry, A* planning and the scripted behaviour policy for world-model data.

Pure numpy/scipy, no Isaac imports, so it can be tested on its own:
    python scenegen.py            # samples a few scenes and runs the planner

Conventions (match sandbox/sim.js):
  * ENU world frame, metres. Obstacles are vertical boxes from z=0 to OBS_H, stored as rows
    (cx, cy, hx, hy, yaw) with half extents hx, hy.
  * Action a (5,) = [v_body / LAMBDA (3), cos(yaw_cmd), sin(yaw_cmd)], v_body in the yaw frame,
    yaw_cmd absolute world yaw.
  * Observation o^s (15,) = [n_goal^b (3), d_xy, d_z, v^b (3), omega^b (3), q_wxyz (4)].
"""
import heapq

import numpy as np
from scipy import ndimage

LAMBDA = np.array([1.5, 1.5, 0.5])
OBS_H = 3.5
Z_GOAL, Z_MIN, Z_MAX = 1.5, 0.3, 3.2
GOAL_R, COLL_R = 1.0, 0.3
HFOV = np.deg2rad(87.0)
D_MIN, D_MAX = 0.6, 6.0
ARENA = 10.0          # obstacles live in [-ARENA, ARENA]^2
EXTENT = 12.0         # maps cover [-EXTENT, EXTENT]^2 (local frame of one environment)
MAX_BOXES = 48

# ego-centric occupancy grid: 64 x 64 cells of 0.15 m, row 0 = 4.8 m ahead, col 0 = 4.8 m left
EGO_N, EGO_RES = 64, 0.15
_c = (np.arange(EGO_N) - (EGO_N - 1) / 2) * EGO_RES
EGO_FWD = -_c[:, None] * np.ones((1, EGO_N))      # (64, 64) forward offset
EGO_LEFT = -_c[None, :] * np.ones((EGO_N, 1))     # (64, 64) left offset

# world-frame "seen" grid
SEEN_RES = 0.15
SEEN_N = int(round(2 * EXTENT / SEEN_RES))        # 160


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot2(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


# ------------------------------------------------------------------ geometry
def box_sdf(points, boxes):
    """Signed 2D distance from points (..., 2) to each box. Returns (..., B)."""
    p = np.asarray(points, np.float64)[..., None, :] - boxes[:, :2]
    c, s = np.cos(boxes[:, 4]), np.sin(boxes[:, 4])
    lx = c * p[..., 0] + s * p[..., 1]
    ly = -s * p[..., 0] + c * p[..., 1]
    qx, qy = np.abs(lx) - boxes[:, 2], np.abs(ly) - boxes[:, 3]
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return outside + np.minimum(np.maximum(qx, qy), 0)


def min_dist(points, boxes):
    if len(boxes) == 0:
        return np.full(np.shape(points)[:-1], np.inf)
    return box_sdf(points, boxes).min(-1)


def raycast2d(origin, dirs, boxes, t_max):
    """Ray parameter t of the first hit for rays origin + t * dirs (dirs (R, 2), need not be unit).

    origin is (2,) or (R, 2). Rays starting inside a box return 0. Misses return t_max.
    """
    if len(boxes) == 0:
        return np.full(len(dirs), t_max)
    o = np.broadcast_to(origin, dirs.shape)[:, None, :] - boxes[:, :2]       # (R, B, 2)
    c, s = np.cos(boxes[:, 4]), np.sin(boxes[:, 4])
    ox, oy = c * o[..., 0] + s * o[..., 1], -s * o[..., 0] + c * o[..., 1]
    dx = c * dirs[:, None, 0] + s * dirs[:, None, 1]
    dy = -s * dirs[:, None, 0] + c * dirs[:, None, 1]
    eps = 1e-9
    dx = np.where(np.abs(dx) < eps, eps, dx)
    dy = np.where(np.abs(dy) < eps, eps, dy)
    tx1, tx2 = (-boxes[:, 2] - ox) / dx, (boxes[:, 2] - ox) / dx
    ty1, ty2 = (-boxes[:, 3] - oy) / dy, (boxes[:, 3] - oy) / dy
    tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    hit = tmax >= np.maximum(tmin, 0)
    t = np.where(hit, np.maximum(tmin, 0), np.inf).min(-1)
    return np.minimum(t, t_max)


def analytic_depth(cam_pos, R_cam, boxes, res=(80, 48)):
    """Exact distance_to_image_plane for the box world + ground plane (used to validate renders).

    R_cam: 3x3 rotation of the camera body frame (x forward, y left, z up) in world.
    """
    W, H = res
    fx = (W / 2) / np.tan(HFOV / 2)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cam = np.stack([np.ones_like(u), -(u - W / 2) / fx, -(v - H / 2) / fx], -1).reshape(-1, 3)
    d = d_cam @ R_cam.T                                            # forward component in cam = 1
    t_box = raycast2d(cam_pos[:2], d[:, :2], boxes, 1e3)
    z_hit = cam_pos[2] + t_box * d[:, 2]
    t_box = np.where((z_hit >= 0) & (z_hit <= OBS_H), t_box, np.inf)
    t_gnd = np.where(d[:, 2] < -1e-6, -cam_pos[2] / np.minimum(d[:, 2], -1e-6), np.inf)
    return np.minimum(t_box, t_gnd).reshape(H, W)


# ------------------------------------------------------------------ scenes
def _trap_boxes(center, yaw, kind, rng):
    """U or C trap whose opening faces local -x, rotated by yaw."""
    W, D, t = rng.uniform(2.4, 4.0), rng.uniform(2.0, 3.5), 0.3
    local = [
        (D / 2 + t / 2, 0.0, t / 2, W / 2 + t),       # back wall
        (0.0, W / 2 + t / 2, D / 2, t / 2),           # side walls
        (0.0, -(W / 2 + t / 2), D / 2, t / 2),
    ]
    if kind == "C":
        gap = rng.uniform(1.2, 1.8)
        lip = (W - gap) / 2
        for sgn in (1, -1):
            local.append((-D / 2 + t / 2, sgn * (gap / 2 + lip / 2), t / 2, lip / 2))
    R = rot2(yaw)
    out = []
    for x, y, hx, hy in local:
        cx, cy = np.asarray(center) + R @ np.array([x, y])
        out.append((cx, cy, hx, hy, yaw))
    return np.array(out), max(W, D) / 2 + 0.6


class Scene:
    def __init__(self, boxes, start, goal, traps, cavities):
        self.boxes, self.start, self.goal = boxes, np.asarray(start), np.asarray(goal)
        self.traps, self.cavities = traps, cavities  # trap kinds, cavity centres
        self.planner = GridPlanner(boxes)


def sample_scene(rng, need_trap=False, max_tries=60):
    for _ in range(max_tries):
        th = rng.uniform(-np.pi, np.pi)
        start = 8.5 * rng.uniform(0.9, 1.05) * np.array([np.cos(th), np.sin(th)])
        th_g = th + np.pi + rng.uniform(-0.5, 0.5)
        goal = 8.5 * rng.uniform(0.9, 1.05) * np.array([np.cos(th_g), np.sin(th_g)])
        boxes, traps, cavities, disks = [], [], [], []   # disks: (centre, radius) of placed traps

        def far_from_endpoints(b, margin=1.5):
            return min_dist(np.stack([start, goal]), b).min() > margin

        n_trap = max(int(need_trap), rng.choice([0, 1, 2], p=[0.3, 0.45, 0.25]))
        for k in range(n_trap):
            for _ in range(30):
                if k == 0 and rng.random() < 0.7:   # on the way, opening towards the start
                    f = rng.uniform(0.35, 0.65)
                    seg = goal - start
                    c = start + f * seg + rng.uniform(-2, 2) * np.array([-seg[1], seg[0]]) / np.linalg.norm(seg)
                    phi = np.arctan2(*(start - c)[::-1])
                    yaw = phi - np.pi + rng.uniform(-0.7, 0.7)
                else:
                    c = rng.uniform(-ARENA + 2.5, ARENA - 2.5, 2)
                    yaw = rng.uniform(-np.pi, np.pi)
                kind = rng.choice(["U", "C"])
                b, r = _trap_boxes(c, yaw, kind, rng)
                if not far_from_endpoints(b):
                    continue
                if any(np.linalg.norm(c - dc) < r + dr + 0.8 for dc, dr in disks):
                    continue
                boxes.extend(b); traps.append(str(kind)); cavities.append(c); disks.append((c, r))
                break

        for _ in range(rng.integers(1, 4)):                 # free-standing walls
            for _ in range(30):
                L, yaw = rng.uniform(3, 8), rng.uniform(-np.pi, np.pi)
                b = np.array([[*rng.uniform(-ARENA + 1, ARENA - 1, 2), L / 2, 0.15, yaw]])
                if far_from_endpoints(b) and _clear_of(b, boxes, 0.9) and _clear_of_disks(b, disks):
                    boxes.extend(b); break
        for _ in range(rng.integers(6, 17)):                # pillars / cuboids
            for _ in range(30):
                b = np.array([[*rng.uniform(-ARENA, ARENA, 2), *rng.uniform(0.2, 0.5, 2), rng.uniform(-np.pi, np.pi)]])
                if far_from_endpoints(b) and _clear_of(b, boxes, 0.9) and _clear_of_disks(b, disks):
                    boxes.extend(b); break
        boxes = np.array(boxes[:MAX_BOXES])
        scene = Scene(boxes, start, goal, traps, cavities)
        if scene.planner.plan(start, goal) is None:
            continue
        if need_trap and (scene.planner.plan(start, cavities[0]) is None
                          or scene.planner.plan(cavities[0], goal) is None):
            continue
        return scene
    raise RuntimeError("could not sample a feasible scene")


def sample_test_scene(rng, kind, max_tries=60, jitter=False):
    """Fixed-structure evaluation scenes (reproducible with a seeded rng).

    utrap: start in front of a U trap's opening, goal straight behind its back wall (must detour).
    ctrap: start inside a C trap facing its back wall, goal behind that wall (must turn back, find the
           gap and go around). A few pillars are scattered away from the trap and the endpoints.
    utrap_in: like ctrap but with a U trap: start inside facing the back wall, fly out of the opening,
           then go around walls that are behind the drone and out of view.
    jitter=True (training starts): random start yaw offset (+-1 rad) and goal bearing (+-60 deg).
    """
    assert kind in ("utrap", "ctrap", "utrap_in"), kind
    for _ in range(max_tries):
        c = rng.uniform(-1.5, 1.5, 2)
        yaw = rng.uniform(-np.pi, np.pi)
        trap, r = _trap_boxes(c, yaw, "C" if kind == "ctrap" else "U", rng)
        R = rot2(yaw)
        gb = rng.uniform(-np.pi / 3, np.pi / 3) if jitter else 0.0
        if kind == "utrap":
            start, goal, start_yaw = c + R @ np.array([-(r + 4.0), 0.0]), c + R @ np.array([r + 4.0, 0.0]), yaw
        else:
            start = c.copy()
            goal = c + R @ ((r + 5.0) * np.array([np.cos(gb), np.sin(gb)]))
            start_yaw = yaw + (rng.uniform(-1.0, 1.0) if jitter else 0.0)
        if np.abs(np.stack([start, goal])).max() > ARENA - 0.5:
            continue
        boxes = list(trap)
        for _ in range(rng.integers(4, 9)):
            for _ in range(30):
                b = np.array([[*rng.uniform(-ARENA, ARENA, 2), *rng.uniform(0.2, 0.5, 2), rng.uniform(-np.pi, np.pi)]])
                if (min_dist(np.stack([start, goal]), b).min() > 1.5 and _clear_of(b, boxes, 0.9)
                        and np.linalg.norm(b[0, :2] - c) > r + 1.5):
                    boxes.append(b[0]); break
        scene = Scene(np.array(boxes), start, goal, [kind], [c])
        if scene.planner.plan(start, goal) is None:
            continue
        scene.start_yaw = start_yaw
        return scene
    raise RuntimeError("could not sample a test scene")


def _clear_of(b, boxes, gap):
    """True if the corners/centre of new box b keep `gap` metres from existing boxes."""
    if len(boxes) == 0:
        return True
    cx, cy, hx, hy, yaw = b[0]
    R = rot2(yaw)
    pts = [np.array([cx, cy]) + R @ np.array([sx * hx, sy * hy]) for sx in (-1, 0, 1) for sy in (-1, 0, 1)]
    return min_dist(np.array(pts), np.array(boxes)).min() > gap


def _clear_of_disks(b, disks):
    return all(np.linalg.norm(b[0, :2] - c) > r + max(b[0, 2], b[0, 3]) for c, r in disks)


# ------------------------------------------------------------------ planning
class GridPlanner:
    RES, INFL, SOFT = 0.2, 0.45, 1.0

    def __init__(self, boxes):
        self.boxes = boxes
        self.n = int(round(2 * EXTENT / self.RES))
        xs = -EXTENT + (np.arange(self.n) + 0.5) * self.RES
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        self.sdf = min_dist(np.stack([X, Y], -1), boxes) if len(boxes) else np.full(X.shape, 99.0)
        self.blocked = self.sdf < self.INFL
        self.cost = 1.0 + 4.0 * np.clip(self.SOFT - self.sdf, 0, None)

    def to_cell(self, p):
        return tuple(np.clip(((np.asarray(p) + EXTENT) / self.RES).astype(int), 0, self.n - 1))

    def to_xy(self, c):
        return -EXTENT + (np.asarray(c) + 0.5) * self.RES

    def plan(self, a, b):
        s, g = self.to_cell(a), self.to_cell(b)
        blocked = self.blocked.copy()
        for c in (s, g):  # allow leaving / reaching cells near obstacles
            i, j = c
            blocked[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3] &= self.sdf[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3] < 0.2
        if blocked[g]:
            return None
        nbrs = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        gs = {s: 0.0}; came = {}; heap = [(0.0, s)]; closed = set()
        while heap:
            _, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            if cur == g:
                break
            closed.add(cur)
            for di, dj, w in nbrs:
                nb = (cur[0] + di, cur[1] + dj)
                if not (0 <= nb[0] < self.n and 0 <= nb[1] < self.n) or blocked[nb]:
                    continue
                ng = gs[cur] + w * self.cost[nb]
                if ng < gs.get(nb, np.inf):
                    gs[nb] = ng; came[nb] = cur
                    heapq.heappush(heap, (ng + np.hypot(nb[0] - g[0], nb[1] - g[1]), nb))
        if g not in came and g != s:
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(came[cells[-1]])
        pts = self.to_xy(np.array(cells[::-1]))
        pts[0], pts[-1] = a, b
        return self._shortcut(pts)

    def _shortcut(self, pts):
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._line_free(pts[i], pts[j]):
                j -= 1
            out.append(pts[j]); i = j
        return np.array(out)

    def _line_free(self, a, b):
        n = max(int(np.linalg.norm(b - a) / 0.1), 2)
        seg = a + np.linspace(0, 1, n)[:, None] * (b - a)
        return min_dist(seg, self.boxes).min() > self.INFL + 0.1 if len(self.boxes) else True


def lookahead_point(path, pos, dist=1.2):
    """Point `dist` metres ahead of the projection of pos on the polyline."""
    best, best_d, best_s = 0, np.inf, 0.0
    seglen = np.linalg.norm(np.diff(path, axis=0), axis=1)
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        t = np.clip(np.dot(pos - a, b - a) / max(seglen[k] ** 2, 1e-9), 0, 1)
        d = np.linalg.norm(a + t * (b - a) - pos)
        if d < best_d:
            best, best_d, best_s = k, d, t * seglen[k]
    s = best_s + dist
    k = best
    while k < len(path) - 1 and s > seglen[k]:
        s -= seglen[k]; k += 1
    if k >= len(path) - 1:
        return path[-1]
    return path[k] + (path[k + 1] - path[k]) * s / max(seglen[k], 1e-9)


# ------------------------------------------------------------------ behaviour policy
class Behaviour:
    """Scripted pilot: A* + pure pursuit with OU action noise.

    modes: "goal"    plan straight to the goal (with noise)
           "explore" visit 2-4 random free waypoints with a wandering yaw offset, then the goal
           "trap"    fly into a trap cavity, look around, then leave for the goal
    glance_rate > 0 (any mode): per step, the chance to start a glance, i.e. look 90-150 deg away from the flight
    direction for 1-4.5 s and back, so that seen obstacles leave the view and return (memory-rich data, kin_v2)
    """

    def __init__(self, scene, mode, rng, glance_rate=0.0):
        self.scene, self.mode, self.rng = scene, mode, rng
        self.glance_rate, self.glance_left, self.glance_off = glance_rate, 0, 0.0
        self.speed = rng.uniform(0.8, 1.5)
        self.sigma = rng.uniform(0.1, 0.35)
        self.z_ref = rng.uniform(1.2, 1.8)
        self.noise = np.zeros(3)
        self.yaw_off, self.yaw_off_sigma = 0.0, (0.6 if mode == "explore" else 0.1)
        self.look_timer = 0
        targets = []
        if mode == "explore":
            for _ in range(rng.integers(2, 5)):
                for _ in range(50):
                    p = rng.uniform(-ARENA + 1, ARENA - 1, 2)
                    if min_dist(p[None], scene.boxes)[0] > 1.0:
                        targets.append(p); break
        elif mode == "trap":
            targets.append(scene.cavities[0])
        targets.append(scene.goal)
        self.targets = targets
        self.path = None

    def _replan(self, pos):
        while self.targets:
            path = self.scene.planner.plan(pos, self.targets[0])
            if path is not None:
                self.path = path
                return
            self.targets.pop(0)   # unreachable intermediate target, skip it
        self.path = np.array([pos, pos])

    def act(self, pos, yaw, t):
        pos2 = pos[:2]
        if self.path is None:
            self._replan(pos2)
        tgt = self.targets[0] if self.targets else self.scene.goal
        if len(self.targets) > 1 and np.linalg.norm(pos2 - tgt) < 0.8:
            self.targets.pop(0)
            if self.mode == "trap":
                self.look_timer = 30          # 1.5 s look-around inside the cavity
            self._replan(pos2)
        if self.look_timer > 0:
            self.look_timer -= 1
            v_xy = np.zeros(2)
            yaw_cmd = yaw + 0.9               # turn in place
        else:
            la = lookahead_point(self.path, pos2)
            d = la - pos2
            clear = min_dist(pos2[None], self.scene.boxes)[0] if len(self.scene.boxes) else 9.0
            v = self.speed * np.clip((clear - 0.35) / 1.0, 0.25, 1.0)
            if self.glance_left == 0 and self.rng.random() < self.glance_rate:
                self.glance_left = int(self.rng.integers(20, 90))
                self.glance_off = self.rng.choice([-1, 1]) * self.rng.uniform(np.deg2rad(90), np.deg2rad(150))
            if self.glance_left > 0:
                self.glance_left -= 1
                v *= 0.6
            v_xy = v * d / max(np.linalg.norm(d), 1e-6)
            if np.linalg.norm(d) < 1e-3:
                v_xy = np.zeros(2)
            self.yaw_off = 0.97 * self.yaw_off + self.yaw_off_sigma * 0.2 * self.rng.standard_normal()
            self.yaw_off = np.clip(self.yaw_off, -0.9, 0.9)
            heading = np.arctan2(v_xy[1], v_xy[0]) if np.linalg.norm(v_xy) > 0.05 else yaw
            yaw_cmd = heading + self.yaw_off + (self.glance_off if self.glance_left > 0 else 0.0)
        vz = np.clip(1.5 * (self.z_ref + 0.2 * np.sin(0.3 * t) - pos[2]), -0.5, 0.5)
        c, s = np.cos(yaw), np.sin(yaw)
        v_body = np.array([c * v_xy[0] + s * v_xy[1], -s * v_xy[0] + c * v_xy[1], vz])
        self.noise = 0.85 * self.noise + self.sigma * np.sqrt(1 - 0.85 ** 2) * self.rng.standard_normal(3)
        clear = min_dist(pos2[None], self.scene.boxes)[0] if len(self.scene.boxes) else 9.0
        k_noise = np.clip((clear - 0.35) / 1.0, 0.15, 1.0)   # calmer near obstacles
        a_v = np.clip(v_body / LAMBDA + k_noise * self.noise, -1, 1)
        yaw_cmd = wrap(yaw_cmd)
        return np.array([*a_v, np.cos(yaw_cmd), np.sin(yaw_cmd)], np.float32)


# ------------------------------------------------------------------ observations / labels
def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def obs15(pos, q, vel_w, angvel_w, goal_xy):
    yaw = yaw_of(q)
    c, s = np.cos(yaw), np.sin(yaw)
    g = np.array([goal_xy[0] - pos[0], goal_xy[1] - pos[1], Z_GOAL - pos[2]])
    n = max(np.linalg.norm(g), 1e-6)
    ng = np.array([(c * g[0] + s * g[1]) / n, (-s * g[0] + c * g[1]) / n, g[2] / n])
    vb = np.array([c * vel_w[0] + s * vel_w[1], -s * vel_w[0] + c * vel_w[1], vel_w[2]])
    omega_b = quat_to_rot(q).T @ angvel_w
    return np.concatenate([ng, [np.hypot(g[0], g[1]), g[2]], vb, omega_b, q]).astype(np.float32)


def ego_points(pos, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    x = pos[0] + EGO_FWD * c - EGO_LEFT * s
    y = pos[1] + EGO_FWD * s + EGO_LEFT * c
    return np.stack([x, y], -1)


N_RAYS = 160
RAY_ANG = np.linspace(-HFOV / 2, HFOV / 2, N_RAYS)


def visibility(points, pos, yaw, ray_range):
    """Cells whose centre is inside the 2D field of view and not occluded (within D_MAX)."""
    rel = points - pos[:2]
    r = np.hypot(rel[..., 0], rel[..., 1])
    bearing = wrap(np.arctan2(rel[..., 1], rel[..., 0]) - yaw)
    inside = np.abs(bearing) <= HFOV / 2
    idx = np.clip(np.round((bearing + HFOV / 2) / HFOV * (N_RAYS - 1)).astype(int), 0, N_RAYS - 1)
    return inside & (r <= np.minimum(ray_range[idx] + EGO_RES, D_MAX))


class SeenMap:
    """World-frame grid of cells that have been inside the camera's (2D) view this episode."""

    def __init__(self):
        self.grid = np.zeros((SEEN_N, SEEN_N), bool)
        xs = -EXTENT + (np.arange(SEEN_N) + 0.5) * SEEN_RES
        self.X, self.Y = np.meshgrid(xs, xs, indexing="ij")

    def update(self, pos, yaw, ray_range):
        w = int(np.ceil((D_MAX + 0.3) / SEEN_RES))
        ci, cj = ((pos[:2] + EXTENT) / SEEN_RES).astype(int)
        i0, i1 = max(ci - w, 0), min(ci + w + 1, SEEN_N)
        j0, j1 = max(cj - w, 0), min(cj + w + 1, SEEN_N)
        if i0 >= i1 or j0 >= j1:
            return
        pts = np.stack([self.X[i0:i1, j0:j1], self.Y[i0:i1, j0:j1]], -1)
        self.grid[i0:i1, j0:j1] |= visibility(pts, pos, yaw, ray_range)

    def sample(self, points):
        ij = np.floor((points + EXTENT) / SEEN_RES).astype(int)
        ok = (ij >= 0).all(-1) & (ij < SEEN_N).all(-1)
        ij = np.clip(ij, 0, SEEN_N - 1)
        return self.grid[ij[..., 0], ij[..., 1]] & ok


def labels(pos, yaw, boxes, seen_map):
    """Ray ranges, ego occupancy, ego seen-so-far mask and ego visible-now mask."""
    dirs = np.stack([np.cos(yaw + RAY_ANG), np.sin(yaw + RAY_ANG)], -1)
    ray_range = raycast2d(pos[:2], dirs, boxes, D_MAX + 1.0)
    seen_map.update(pos, yaw, ray_range)
    pts = ego_points(pos, yaw)
    occ = min_dist(pts, boxes) <= 0 if len(boxes) else np.zeros(pts.shape[:2], bool)
    vis = visibility(pts, pos, yaw, ray_range)
    seen = seen_map.sample(pts) | vis
    return ray_range, occ, seen, vis


def reward_terms(pos, goal_xy, d_xy_prev, boxes):
    d_xy = np.hypot(goal_xy[0] - pos[0], goal_xy[1] - pos[1])
    dob = min_dist(pos[None, :2], boxes)[0] if len(boxes) else 9.0
    terms = {"prog": d_xy_prev - d_xy, "safety": -max(0.0, 1 - dob) ** 2,
             "height": -(Z_GOAL - pos[2]) ** 2, "goal": 0.0, "coll": 0.0}
    status = None
    if dob < COLL_R:
        terms["coll"], status = -5.0, "collision"
    elif np.hypot(d_xy, Z_GOAL - pos[2]) < GOAL_R:
        terms["goal"], status = 100.0, "goal"
    elif not (Z_MIN <= pos[2] <= Z_MAX):
        status = "height"
    r = 10 * terms["prog"] + 5 * terms["safety"] + 2 * terms["height"] + terms["goal"] + terms["coll"]
    return r, d_xy, status, dob


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    for k in range(6):
        t0 = time.time()
        sc = sample_scene(rng, need_trap=(k % 2 == 1))
        path = sc.planner.plan(sc.start, sc.goal)
        print(f"scene {k}: {len(sc.boxes)} boxes, traps={sc.traps}, path pts={len(path)}, "
              f"len={np.linalg.norm(np.diff(path, axis=0), axis=1).sum():.1f} m, {time.time() - t0:.2f}s")
