"""Gymnasium environment: Unitree G1 walking to a goal in MuJoCo.

The G1 model comes from MuJoCo Menagerie (MJX version, simple colliders, realistic PD gains), see the g1/ folder.
The policy controls the 12 leg joints. The waist and arms are held at the keyframe "home" pose.
Action in [-1, 1] is converted to a target joint angle around the standing pose LEG_Q_DEF, then MuJoCo's PD produces torque.
The policy runs at 50 Hz (timestep 0.004 s, frame_skip 5).

Hai scene:
  "field":  random terrain 30 m x 30 m, goal changes continuously (free-walking training, with pushes).
  "course": corridor 24 m x 6 m with challenges (slope, stairs, gaps, stepping stones, low obstacles,
            balance beam, rough ground). The robot starts at one end, the goal is at the other, see terrains.py.
"""
import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.ndimage import gaussian_filter

import terrains

_HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = {
    "field": os.path.join(_HERE, "g1", "scene_g1.xml"),
    "course": os.path.join(_HERE, "g1", "scene_g1_course.xml"),
}
COURSE_BASE = 0.5  # base ground height (m) of the corridor in the hfield, leaving room for deep pits

# Leg joint order by G1 actuator: left leg then right leg,
# each leg has hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll.
N_LEG = 12
LEG_Q_DEF = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2)       # matches keyframe "home"
LEG_ACT_SCALE = np.array([0.5, 0.3, 0.3, 0.6, 0.4, 0.2] * 2)

ROOT_BODY = "pelvis"
PAYLOAD_BODY = "torso_link"
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_GEOMS = (
    ("left_foot1_collision", "left_foot2_collision", "left_foot3_collision"),
    ("right_foot1_collision", "right_foot2_collision", "right_foot3_collision"),
)
TERRAIN_GEOM = "terrain"

# Fall threshold based on pelvis height relative to the feet (upright G1 is about 0.75 m)
FALL_HEIGHT = 0.45
FULL_HEIGHT = 0.70
MIN_CLEARANCE = 0.30   # pelvis this far below the ground right under it counts as a fall (fell into a pit)

# Height observation grid, in the robot's heading frame (m): front/back x left/right
SCAN_FWD = np.linspace(-0.2, 1.0, 7)
SCAN_LAT = np.linspace(-0.3, 0.3, 5)

DEFAULT_REWARD_SCALES = dict(
    vel=1.0,       # track the desired velocity toward the goal
    heading=0.3,   # torso faces the goal
    upright=0.2,   # torso upright
    height=0.2,    # keep hip height
    alive=0.5,     # small reward for each step alive
    action=0.01,   # penalty on squared action (energy)
    smooth=0.02,   # penalty on action change between steps (jerk)
    ang_vel=0.02,  # penalty on torso swaying about the horizontal axes
    vz=0.05,       # penalty on bouncing up and down
    reach=5.0,     # reward for reaching the goal
    fall=5.0,      # penalty for falling
)


class BipedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scene="field",
        challenge="mixed",
        challenge_level=0.0,
        height_scan=False,
        scan_noise=0.0,
        frame_skip=5,
        max_steps=None,
        target_speed=1.0,
        target_range=(4.0, 8.0),
        target_angle=0.6,
        reach_radius=0.5,
        reward_scales=None,
        randomize=True,
        push_max=0.0,
        roughness=0.0,
        push_interval=(2.0, 5.0),
        push_duration=0.1,
        on_reach="resample",
        kp_scale=1.0,
    ):
        super().__init__()
        if scene not in SCENES:
            raise ValueError(f"scene must be one of {list(SCENES)}")
        self.scene = scene
        self.challenge = challenge
        self.challenge_level = float(challenge_level)
        self.height_scan = height_scan
        self.scan_noise = scan_noise
        self.model = mujoco.MjModel.from_xml_path(SCENES[scene])
        self.data = mujoco.MjData(self.model)
        m = self.model

        if kp_scale != 1.0:  # raise/lower the PD stiffness of every actuator (kp is in gainprm[0] and biasprm[1])
            m.actuator_gainprm[:, 0] *= kp_scale
            m.actuator_biasprm[:, 1] *= kp_scale

        self.frame_skip = frame_skip
        self.dt = m.opt.timestep * frame_skip
        self.max_steps = max_steps if max_steps is not None else (1500 if scene == "course" else 1000)
        self.target_speed = target_speed
        self.target_range = target_range
        self.target_angle = target_angle
        self.reach_radius = reach_radius
        self.randomize = randomize
        self.push_max = push_max
        self.roughness = roughness
        self.push_interval = push_interval
        self.push_duration = push_duration
        self.on_reach = on_reach  # "resample" | "terminate" | "ignore" (scene course always defaults to "terminate")
        self.scales = dict(DEFAULT_REWARD_SCALES)
        if reward_scales:
            unknown = set(reward_scales) - set(self.scales)
            if unknown:
                raise ValueError(f"invalid reward_scales: {unknown}")
            self.scales.update(reward_scales)

        obj = mujoco.mjtObj
        self.torso_id = mujoco.mj_name2id(m, obj.mjOBJ_BODY, ROOT_BODY)  # the name "torso" is kept for compatibility
        self.payload_id = mujoco.mj_name2id(m, obj.mjOBJ_BODY, PAYLOAD_BODY)
        self.foot_bodies = [mujoco.mj_name2id(m, obj.mjOBJ_BODY, n) for n in FOOT_BODIES]
        self.foot_geom_sets = [[mujoco.mj_name2id(m, obj.mjOBJ_GEOM, n) for n in s] for s in FOOT_GEOMS]
        self.foot_geoms = [g for s in self.foot_geom_sets for g in s]
        self.terrain_geom = mujoco.mj_name2id(m, obj.mjOBJ_GEOM, TERRAIN_GEOM)

        # Per-actuator info (29 joints): qpos address, dof, limits
        jids = m.actuator_trnid[:, 0]
        self.qadr_all = m.jnt_qposadr[jids]
        self.dadr_all = m.jnt_dofadr[jids]
        self.qadr = self.qadr_all[:N_LEG]
        self.dadr = self.dadr_all[:N_LEG]
        self.jrange = m.jnt_range[jids[:N_LEG]]
        self.nj = N_LEG
        key = m.key("home")
        self.home_qpos = key.qpos.copy()
        self.home_ctrl = key.ctrl.copy()
        assert np.allclose(self.home_ctrl[:N_LEG], LEG_Q_DEF), "LEG_Q_DEF must match keyframe home"
        self.total_mass = float(m.body_subtreemass[self.torso_id])

        self._mass0 = m.body_mass.copy()
        self._inertia0 = m.body_inertia.copy()
        self._pair_fric0 = m.pair_friction.copy()
        self._hf_rows = int(m.hfield_nrow[0])
        self._hf_cols = int(m.hfield_ncol[0])
        self._hf_sx, self._hf_sy, self._hf_z = (float(v) for v in m.hfield_size[0, :3])
        self._xs = np.linspace(-self._hf_sx, self._hf_sx, self._hf_cols)
        self._ys = np.linspace(-self._hf_sy, self._hf_sy, self._hf_rows)
        self._course = None

        # height observation grid
        fx, ly = np.meshgrid(SCAN_FWD, SCAN_LAT, indexing="ij")
        self._scan_offsets = np.stack([fx.ravel(), ly.ravel()], axis=1)  # (35, 2): (front, left)

        obs_dim = 3 + 3 + 3 + 3 * self.nj + 3 + 2 + (len(self._scan_offsets) if height_scan else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (self.nj,), np.float32)

        self._prev_action = np.zeros(self.nj)
        self._target = np.zeros(2)
        self._steps = 0
        self.current_push = np.zeros(3)  # force currently applied, used to draw the arrow when recording video

    # ------------------------------------------------------------------ difficulty config
    def set_difficulty(self, push_max, terrain, speed=None):
        """Called via VecEnv.env_method to raise difficulty per the curriculum. Takes effect from the next reset.

        `terrain`: roughness (m) for scene "field", or challenge difficulty [0, 1] for scene "course".
        `speed`: commanded walking speed (m/s); None keeps the current one.
        """
        self.push_max = float(push_max)
        if self.scene == "course":
            self.challenge_level = float(terrain)
        else:
            self.roughness = float(terrain)
        if speed is not None:
            self.target_speed = float(speed)

    # ------------------------------------------------------------------ terrain
    def _hf_array(self):
        return self.model.hfield_data[: self._hf_rows * self._hf_cols].reshape(self._hf_rows, self._hf_cols)

    def _build_field_terrain(self, roughness):
        rows, cols = self._hf_rows, self._hf_cols
        if roughness <= 0:
            arr = np.full((rows, cols), 0.5)
        else:
            rng = self.np_random
            fine = gaussian_filter(rng.standard_normal((rows, cols)), 2.5)
            coarse = gaussian_filter(rng.standard_normal((rows, cols)), 8.0)
            noise = 0.5 * fine / fine.std() + 0.5 * coarse / coarse.std()
            noise = np.clip(noise / 2.0, -1.0, 1.0)  # amplitude about 2 sigma
            r = np.hypot(self._xs[None, :], self._ys[:, None])
            mask = np.clip((r - 1.5) / 2.0, 0.0, 1.0)  # flat start zone of radius 1.5 m
            arr = 0.5 + (roughness / self._hf_z) * noise * mask
        self.model.hfield_data[:] = arr.ravel()

    def _build_course_terrain(self, kind, level):
        H, info = terrains.build_course(self._xs, self._ys, kind, level, self.np_random)
        arr = (COURSE_BASE + H) / self._hf_z
        assert arr.min() >= 0.0 and arr.max() <= 1.0, "height exceeds the hfield range, increase size[2] in the scene"
        self.model.hfield_data[:] = arr.ravel()
        self._course = info

    def _ground_z_vec(self, xs, ys):
        """Ground height at the points (xs, ys), bilinearly interpolated from the heightfield data."""
        data = self._hf_array()
        cx = np.clip((np.asarray(xs, dtype=float) / (2 * self._hf_sx) + 0.5) * (self._hf_cols - 1), 0, self._hf_cols - 1.001)
        cy = np.clip((np.asarray(ys, dtype=float) / (2 * self._hf_sy) + 0.5) * (self._hf_rows - 1), 0, self._hf_rows - 1.001)
        j, i = cx.astype(int), cy.astype(int)
        fx, fy = cx - j, cy - i
        v = (data[i, j] * (1 - fx) * (1 - fy) + data[i, j + 1] * fx * (1 - fy)
             + data[i + 1, j] * (1 - fx) * fy + data[i + 1, j + 1] * fx * fy)
        return v * self._hf_z

    def _ground_z(self, x, y):
        return float(self._ground_z_vec(np.array([x]), np.array([y]))[0])

    def _foot_min_z(self):
        """Lowest point of the foot geoms (capsule, sphere or box)."""
        zs = []
        for gid in self.foot_geoms:
            R = self.data.geom_xmat[gid].reshape(3, 3)
            p = self.data.geom_xpos[gid]
            s = self.model.geom_size[gid]
            t = self.model.geom_type[gid]
            if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                zs.append(p[2] - (s[0] + abs(R[2, 2]) * s[1]))
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                zs.append(p[2] - s[0])
            else:
                zs.append(p[2] - np.sum(np.abs(R[2, :]) * s))
        return min(zs)

    def _foot_contacts(self):
        d = self.data
        flags = [0.0, 0.0]
        n = d.ncon
        if n:
            g1 = d.contact.geom1[:n]
            g2 = d.contact.geom2[:n]
            on_terrain = (g1 == self.terrain_geom) | (g2 == self.terrain_geom)
            other = np.where(g1 == self.terrain_geom, g2, g1)
            for k, ids in enumerate(self.foot_geom_sets):
                flags[k] = float(np.any(on_terrain & np.isin(other, ids)))
        return np.array(flags)

    # ------------------------------------------------------------------ goal
    def _sample_target(self, base_xy, angle=None, dist=None):
        rng = self.np_random
        for _ in range(20):
            a = rng.uniform(-self.target_angle, self.target_angle) if angle is None else angle
            dd = rng.uniform(*self.target_range) if dist is None else dist
            t = base_xy + dd * np.array([np.cos(a), np.sin(a)])
            if np.abs(t).max() < 12.0:
                return t
        return np.zeros(2)

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}
        m, d = self.model, self.data
        rng = self.np_random
        course = self.scene == "course"

        self.on_reach_ep = opts.get("on_reach", "terminate" if course else self.on_reach)
        self.randomize_ep = opts.get("randomize", self.randomize)
        self._ep_push_max = opts.get("push_max", self.push_max)
        self._fixed_push = opts.get("push_impulse", None)  # N*s, a single push (evaluation)
        self._fixed_push_time = opts.get("push_time", 3.0)
        self._fixed_push_angle = opts.get("push_angle", None)
        payload = float(opts.get("payload", 0.0))           # kg added to the upper body
        friction = opts.get("friction", None)               # fixed friction multiplier, overrides randomization

        mujoco.mj_resetData(m, d)
        m.body_mass[:] = self._mass0
        m.body_inertia[:] = self._inertia0
        m.pair_friction[:] = self._pair_fric0
        if self.randomize_ep:
            s = rng.uniform(0.85, 1.15)
            m.body_mass[:] *= s
            m.body_inertia[:] *= s
            m.pair_friction[:, :2] = self._pair_fric0[:, :2] * rng.uniform(0.7, 1.3)
        if payload > 0:
            k = 1.0 + payload / m.body_mass[self.payload_id]
            m.body_mass[self.payload_id] *= k
            m.body_inertia[self.payload_id] *= k
        if friction is not None:
            m.pair_friction[:, :2] = self._pair_fric0[:, :2] * float(friction)
        mujoco.mj_setConst(m, d)
        self.total_mass = float(m.body_subtreemass[self.torso_id])

        if course:
            self._build_course_terrain(opts.get("challenge", self.challenge),
                                       opts.get("challenge_level", self.challenge_level))
            start_xy = np.array([self._course["start_x"], 0.0])
        else:
            self._build_field_terrain(opts.get("roughness", self.roughness))
            start_xy = np.zeros(2)

        noise = (1.0 if self.randomize_ep else 0.0)
        d.qpos[:] = self.home_qpos
        d.qpos[0:2] = start_xy
        d.qpos[self.qadr] += rng.uniform(-0.03, 0.03, self.nj) * noise
        d.qvel[:] = rng.uniform(-0.05, 0.05, m.nv) * noise
        d.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(m, d)
        d.qpos[2] += self._ground_z(*start_xy) + 0.005 - self._foot_min_z()
        mujoco.mj_forward(m, d)

        if course:
            self._target = np.array([self._course["goal_x"], 0.0])
        else:
            self._target = self._sample_target(
                start_xy, angle=opts.get("target_angle_fixed"), dist=opts.get("target_dist")
            )
        self._set_target_marker()
        self._prev_action = np.zeros(self.nj)
        self._steps = 0
        self._n_reached = 0
        self._energy = 0.0
        self._path = 0.0
        self._push_end = -1.0
        self._push_vec = np.zeros(3)
        self.current_push = np.zeros(3)
        self._next_push = (
            self._fixed_push_time if self._fixed_push is not None
            else rng.uniform(*self.push_interval)
        )
        return self._obs(), {}

    # ------------------------------------------------------------------ push
    def _update_push(self):
        d = self.data
        t = d.time
        if self._push_end > 0 and t >= self._push_end:
            d.xfrc_applied[self.torso_id, :] = 0.0
            self._push_end = -1.0
            self.current_push = np.zeros(3)
        if self._push_end < 0 and t >= self._next_push:
            rng = self.np_random
            if self._fixed_push is not None:
                impulse = float(self._fixed_push)
                ang = self._fixed_push_angle if self._fixed_push_angle is not None else rng.uniform(-np.pi, np.pi)
                self._next_push = np.inf
            else:
                impulse = rng.uniform(0.0, self._ep_push_max) if self._ep_push_max > 0 else 0.0
                ang = rng.uniform(-np.pi, np.pi)
                self._next_push = t + rng.uniform(*self.push_interval)
            if impulse > 0:
                f = impulse / self.push_duration
                self._push_vec = np.array([f * np.cos(ang), f * np.sin(ang), 0.0])
                d.xfrc_applied[self.torso_id, :3] = self._push_vec
                self._push_end = t + self.push_duration
                self.current_push = self._push_vec.copy()

    # ------------------------------------------------------------------ observation
    def _heading(self):
        R = self.data.xmat[self.torso_id].reshape(3, 3)
        fwd = R[:2, 0]
        n = np.linalg.norm(fwd)
        return fwd / n if n > 1e-6 else np.array([1.0, 0.0])

    def _height_scan(self):
        """Ground height at the grid points around the robot, relative to the ground right under the pelvis (m, clipped to [-1, 1])."""
        d = self.data
        h = self._heading()
        left = np.array([-h[1], h[0]])
        pts = d.qpos[:2] + self._scan_offsets[:, :1] * h + self._scan_offsets[:, 1:] * left
        z = self._ground_z_vec(pts[:, 0], pts[:, 1]) - self._ground_z(*d.qpos[:2])
        if self.scan_noise > 0:
            z = z + self.np_random.normal(0.0, self.scan_noise, z.shape)
        return np.clip(z, -1.0, 1.0)

    def _obs(self):
        d = self.data
        R = d.xmat[self.torso_id].reshape(3, 3)
        gravity_local = -R[2, :]               # gravity vector in the pelvis frame
        ang_vel = d.qvel[3:6]                  # free joint: angular velocity in the body frame
        lin_vel = R.T @ d.qvel[0:3]
        qj = d.qpos[self.qadr] - LEG_Q_DEF
        dqj = d.qvel[self.dadr]
        to_t = self._target - d.qpos[:2]
        dist = np.linalg.norm(to_t)
        h = self._heading()
        dir_local = np.array([h[0] * to_t[0] + h[1] * to_t[1], -h[1] * to_t[0] + h[0] * to_t[1]]) / max(dist, 1e-6)
        parts = [
            gravity_local, ang_vel, lin_vel, qj, dqj * 0.1, self._prev_action,
            dir_local, [min(dist, 10.0) / 10.0], self._foot_contacts(),
        ]
        if self.height_scan:
            parts.append(self._height_scan())
        return np.concatenate(parts)

    def _set_target_marker(self):
        z = self._ground_z(*self._target) + 0.3
        self.data.mocap_pos[0] = [self._target[0], self._target[1], z]

    # ------------------------------------------------------------------ step
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        d, m = self.data, self.model
        ctrl = self.home_ctrl.copy()
        ctrl[:N_LEG] = np.clip(LEG_Q_DEF + a * LEG_ACT_SCALE, self.jrange[:, 0], self.jrange[:, 1])
        d.ctrl[:] = ctrl
        pos0 = d.qpos[:2].copy()
        energy = 0.0
        for _ in range(self.frame_skip):
            self._update_push()
            mujoco.mj_step(m, d)
            energy += float(np.sum(np.abs(d.qfrc_actuator[self.dadr_all] * d.qvel[self.dadr_all]))) * m.opt.timestep
        self._steps += 1
        self._energy += energy
        self._path += float(np.linalg.norm(d.qpos[:2] - pos0))

        R = d.xmat[self.torso_id].reshape(3, 3)
        up_z = R[2, 2]
        foot_z = np.mean([d.xpos[b][2] for b in self.foot_bodies])
        height = d.qpos[2] - foot_z
        clearance = d.qpos[2] - self._ground_z(*d.qpos[:2])
        out_of_bounds = (abs(d.qpos[1]) > self._hf_sy - 0.15 or d.qpos[0] < -self._hf_sx + 0.1
                         or d.qpos[0] > self._hf_sx - 0.1)
        to_t = self._target - d.qpos[:2]
        dist = float(np.linalg.norm(to_t))
        dir_w = to_t / max(dist, 1e-6)
        v_toward = float(d.qvel[0] * dir_w[0] + d.qvel[1] * dir_w[1])
        h = self._heading()
        cos_head = float(h @ dir_w)

        sc = self.scales
        reward = (
            sc["vel"] * np.exp(-4.0 * (self.target_speed - v_toward) ** 2)
            + sc["heading"] * max(0.0, cos_head)
            + sc["upright"] * max(0.0, up_z)
            + sc["height"] * float(np.clip((height - FALL_HEIGHT - 0.05) / (FULL_HEIGHT - FALL_HEIGHT - 0.05), 0.0, 1.0))
            + sc["alive"]
            - sc["action"] * float(np.sum(a ** 2))
            - sc["smooth"] * float(np.sum((a - self._prev_action) ** 2))
            - sc["ang_vel"] * float(np.sum(d.qvel[3:5] ** 2))
            - sc["vz"] * float(d.qvel[2] ** 2)
        )
        smooth_abs = float(np.mean(np.abs(a - self._prev_action)))
        self._prev_action = a

        reached = dist < self.reach_radius
        terminated = False
        success = False
        if reached:
            if self.on_reach_ep == "terminate":
                terminated = True
                success = True
                reward += sc["reach"]
            elif self.on_reach_ep == "resample":
                reward += sc["reach"]
                self._n_reached += 1
                self._target = self._sample_target(d.qpos[:2].copy())
                self._set_target_marker()
        fell = bool(up_z < 0.5 or height < FALL_HEIGHT or clearance < MIN_CLEARANCE or out_of_bounds
                    or not np.all(np.isfinite(d.qpos)))
        if fell and not success:
            terminated = True
            reward -= sc["fall"]
        else:
            fell = False
        truncated = (not terminated) and self._steps >= self.max_steps

        info = {
            "fell": float(fell),
            "success": float(success),
            "n_reached": float(self._n_reached),
            "energy": energy,
            "smooth_abs": smooth_abs,
            "dist": dist,
            "path": self._path,
            "total_energy": self._energy,
            "x": float(d.qpos[0]),
        }
        return self._obs(), float(reward), terminated, truncated, info
