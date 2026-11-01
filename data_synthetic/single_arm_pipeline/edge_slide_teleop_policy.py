"""
edge_slide_teleop_policy.py
===========================
Teleoperator-style scripted policy for ``ib-p1-edge-slide-v1``.

Same strategy as :class:`EdgeSlidePushPolicy` (slide the plate so it overhangs
the +Y table edge, then grasp the overhang from the side and lift), but the
motion is shaped to look like a human demonstration:

* **Plate-aware geometry.** The push target and grasp point are computed from
  the plate's actual randomised half-extents and its current yaw (privileged
  sim info, fine for a scripted demonstrator). The old policy assumed a fixed
  20 cm plate aligned with the edge, which left small plates with no overhang
  and pushed large ones too far.
* **One push.** A single (possibly diagonal) push through the plate centre
  toward a point on the edge inside the arm's comfortable grasp window,
  instead of a push followed by separate lateral correction passes.
* **Smooth motion.** Free-space moves follow min-jerk position/orientation
  profiles tracked with feed-forward + small P feedback, and the commanded
  action is slew-rate limited, so there are no velocity jumps between phases.
  The clearance hop is ~10 cm instead of 30 cm.
* **Phases end on arrival**, not on time-outs (time-outs are only a safety
  net). The gripper closes until the fingers stop moving, then lifts at once.
* **Consistent gripper use.** Closed while pushing (fingers as a paddle),
  opened once in transit to the grasp, closed once to grasp.
* **Per-episode style variation** (speed, clearance height, overhang, grasp
  depth, lift height) drawn from an RNG seeded by the initial scene, so every
  env seed is still exactly reproducible.
* **Recovery.** A missed grasp or a slip during lift opens the gripper, backs
  off and re-grasps; if the overhang is too small it goes back and pushes
  again.
* **Optional DART noise.** ``perturb()`` returns the action with smooth
  (Ornstein-Uhlenbeck) noise for execution while the clean action is
  recorded as the label. Off unless ``dart_sigma > 0``.

Run standalone::

    python data_synthetic/single_arm_pipeline/edge_slide_teleop_policy.py --render --seeds 0 7 14
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from imbench.policy.base_policy import PolicyInterface

# ── Scene constants (ib-p1-edge-slide-v1) ─────────────────────────────
TABLE_Z      = 0.80    # table top
TABLE_EDGE_Y = 0.25    # distance of every table edge from the centre (table 0.5 x 0.5 m)

# Table edges the plate can be slid to. ``yaw`` rotates the +Y edge frame
# (outward normal n=(0,1), tangent t=(1,0)) about world Z onto that edge.
# The -X edge faces the robot base and is not reachable for a side grasp.
EDGES = {
    "+y": 0.0,
    "-y": np.pi,
    "+x": -np.pi / 2,
}

# ── Controller calibration (IK, cartesian_delta, 100 Hz) ──────────────
# Measured per control rate: (m per step per unit action, rad per step per unit action).
CALIB = {
    100: (0.0032, np.radians(1.79)),
    20:  (0.0144, np.radians(7.4)),
}
TRACK_TAU = 0.195   # s, time constant of the position/orientation feedback
PUSH_TAU  = 0.095   # s, extra stiffness along the push
SLEW_RATE = 4.0     # max change of an arm action dim per second
A_MAX     = 0.7     # |action| cap for arm dims

# ── Orientations (xyzw) ───────────────────────────────────────────────
QUAT_DOWN  = np.array([1.0, 0.0, 0.0, 0.0])          # fingers pointing down
QUAT_GRASP = np.array([0.7071068, 0.0, 0.0, 0.7071068])  # approach along -Y, fingers clamp Z

# ── Geometry margins ──────────────────────────────────────────────────
PUSH_Z            = TABLE_Z + 0.012   # eef height while pushing
PUSH_GAP          = 0.025             # gap between fingers and plate before the push
CENTER_MARGIN     = 0.018            # plate centre must stay this far inside the edge
MIN_OVERHANG      = 0.018             # below this, re-push instead of re-grasp
FINGER_EDGE_CLEAR = 0.012             # eef y must be this far past the table edge
GRASP_X_WINDOW    = (-0.04, 0.08)     # comfortable world-x range for the side grasp (+-Y edges)
GRASP_Y_WINDOW    = (-0.08, 0.08)     # comfortable world-y range for the side grasp (+X edge)
PREGRASP_BACKOFF  = 0.07              # pre-grasp stand-off beyond the plate tip

MAX_RECOVERIES = 3


def _min_jerk(tau: float) -> tuple[float, float]:
    """Return (s, ds/dtau) of the min-jerk profile for tau in [0, 1]."""
    tau = float(np.clip(tau, 0.0, 1.0))
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    return s, ds


class _Segment:
    """Min-jerk move from (p0, q0) to (p1, q1) over T seconds."""

    def __init__(self, p0, q0, p1, q1, v_lin, w_ang, t_min=0.35):
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        r0, r1 = Rotation.from_quat(q0), Rotation.from_quat(q1)
        self._slerp = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))
        ang = (r1 * r0.inv()).magnitude()
        self._axis_ang = (r1 * r0.inv()).as_rotvec()
        dist = np.linalg.norm(self.p1 - self.p0)
        # 1.875 = peak/mean velocity ratio of min-jerk: keeps the *peak* at v_lin.
        self.T = max(t_min, 1.875 * dist / v_lin, 1.875 * ang / w_ang)
        self.t = 0.0

    def ref(self):
        tau = self.t / self.T
        s, ds = _min_jerk(tau)
        s = min(max(s, 0.0), 1.0)    # the polynomial can exceed 1 by float error; Slerp rejects that
        pos = self.p0 + s * (self.p1 - self.p0)
        vel = ds / self.T * (self.p1 - self.p0)          # m/s
        quat = self._slerp([s])[0].as_quat()
        omega = ds / self.T * self._axis_ang             # rad/s (world frame)
        return pos, quat, vel, omega

    @property
    def finished(self):
        return self.t >= self.T


class EdgeSlideTeleopPolicy(PolicyInterface):

    # Phases where an operator moves carefully: less wander, a bit less tremor.
    _FINE_PHASES = {"descend_push", "push", "approach_grasp", "close", "lift"}

    def __init__(self, dart_sigma: float = 0.0, human_noise: float = 0.0,
                 verbose: bool = True, control_freq: int = 100,
                 edge_mode: str = "+y", allowed_edges=("+y", "-y")):
        """``edge_mode``: an edge name from :data:`EDGES` (``"+y"`` = original
        behaviour), or ``"closest"`` to pick, per episode, the edge in
        ``allowed_edges`` the plate needs the shortest push to overhang."""
        if edge_mode != "closest" and edge_mode not in EDGES:
            raise ValueError(f"edge_mode must be 'closest' or one of {sorted(EDGES)}")
        if not allowed_edges or any(e not in EDGES for e in allowed_edges):
            raise ValueError(f"allowed_edges must be a non-empty subset of {sorted(EDGES)}")
        self.edge_mode = edge_mode
        self.allowed_edges = tuple(allowed_edges)
        if control_freq not in CALIB:
            raise ValueError(f"no controller calibration for {control_freq} Hz; known: {sorted(CALIB)}")
        self._dt = 1.0 / control_freq
        self._pos_scale, self._rot_scale = CALIB[control_freq]
        self._kp = 1.0 - np.exp(-self._dt / TRACK_TAU)
        self._kpush = 1.0 - np.exp(-self._dt / PUSH_TAU)
        self._slew = SLEW_RATE * self._dt
        self._settle_win = max(3, int(round(0.08 / self._dt)))
        self.dart_sigma = dart_sigma
        self.human_noise = human_noise
        self.verbose = verbose
        self.reset()

    # ── lifecycle ──────────────────────────────────────────────────
    def reset(self, seed: int | None = None):
        self.phase = "start"
        self._seed = seed
        self._rng = None
        self._seg = None
        self._t_phase = 0.0
        self._prev_action = np.zeros(7)
        self._gripper = 1.0
        self._half = None
        self._recoveries = 0
        self._noise = np.zeros(6)
        self._hn_gain = 0.0            # speed-variation state
        self._hn_wander = np.zeros(6)  # slow path-wander state
        self._hn_tremor = np.zeros(6)  # tremor state
        self._grip_hist = []
        self._push = {}
        self._grasp = {}
        self._lift_ref = None
        self._set_edge("+y" if self.edge_mode == "closest" else self.edge_mode)

    # ── edge frame ─────────────────────────────────────────────────
    def _set_edge(self, name: str):
        """World xy = R @ (u, w): u along the edge tangent t, w along the outward normal n."""
        yaw = EDGES[name]
        self.edge = name
        if yaw == 0.0:      # exact identity: +Y reproduces the original policy bit-for-bit
            self._R = np.eye(2)
            self._q_grasp = QUAT_GRASP.copy()
        else:
            c, s_ = np.cos(yaw), np.sin(yaw)
            self._R = np.array([[c, -s_], [s_, c]])
            # A parallel-jaw grasp is symmetric under a 180° roll about the
            # approach axis; take the variant closest to fingers-down so the
            # wrist stays inside its joint limits (the plain 180°-yawed grasp
            # drives Panda joint 6 into its limit on the -Y side).
            down = Rotation.from_quat(QUAT_DOWN)
            a = Rotation.from_euler("z", yaw) * Rotation.from_quat(QUAT_GRASP)
            b = a * Rotation.from_euler("z", np.pi)
            self._q_grasp = min((a, b), key=lambda r: (r * down.inv()).magnitude()).as_quat()
        self._t = self._R[:, 0].copy()
        self._n = self._R[:, 1].copy()
        # Comfortable grasp window along the tangent, in edge coordinates.
        if abs(self._t[0]) > 0.5:
            win = np.array(GRASP_X_WINDOW) * np.sign(self._t[0])
        else:
            win = np.array(GRASP_Y_WINDOW) * np.sign(self._t[1])
        self._u_window = (float(win.min()), float(win.max()))

    def _choose_edge(self, obj_pos, obj_quat):
        """Edge needing the shortest push for the plate's tip to reach it."""
        best, best_gap = None, np.inf
        for name in self.allowed_edges:
            self._set_edge(name)
            gap = TABLE_EDGE_Y - (self._n @ obj_pos[:2] + self._support(obj_quat, self._n))
            if gap < best_gap:
                best, best_gap = name, gap
        self._set_edge(best)
        return best

    def _to_edge(self, xy):
        return self._R.T @ np.asarray(xy, float)

    def _to_world(self, uw):
        return self._R @ np.asarray(uw, float)

    def _advance(self, nxt: str):
        if self.verbose:
            print(f"  [{int(round(self._t_phase / self._dt)):4d} steps]  {self.phase}  →  {nxt}", flush=True)
        self.phase = nxt
        self._t_phase = 0.0
        self._seg = None

    # ── plate geometry ─────────────────────────────────────────────
    def _read_plate_size(self, env):
        task = env
        sim = task._env.sim
        bid = task._find_body_id(sim, "cube")
        for i in range(sim.model.ngeom):
            if sim.model.geom_bodyid[i] == bid:
                return np.array(sim.model.geom_size[i][:3], float)
        raise RuntimeError("plate geom not found")

    def _axes_xy(self, obj_quat):
        m = Rotation.from_quat(obj_quat).as_matrix()
        ax, ay = m[:2, 0], m[:2, 1]
        return ax / (np.linalg.norm(ax) + 1e-9), ay / (np.linalg.norm(ay) + 1e-9)

    def _support(self, obj_quat, d):
        """Half-extent of the plate footprint along unit xy direction d."""
        ax, ay = self._axes_xy(obj_quat)
        return abs(d @ ax) * self._half[0] + abs(d @ ay) * self._half[1]

    def _footprint(self, obj_pos, obj_quat, n=21):
        ax, ay = self._axes_xy(obj_quat)
        u = np.linspace(-self._half[0], self._half[0], n)
        v = np.linspace(-self._half[1], self._half[1], n)
        uu, vv = np.meshgrid(u, v)
        pts = obj_pos[:2] + uu.reshape(-1, 1) * ax + vv.reshape(-1, 1) * ay
        return pts

    def _tip_n(self, obj_pos, obj_quat):
        """Plate tip coordinate along the edge's outward normal."""
        return self._n @ obj_pos[:2] + self._support(obj_quat, self._n)

    # ── motion primitives ──────────────────────────────────────────
    def _track(self, eef_pos, eef_quat, p_ref, q_ref, v_ref=None, w_ref=None):
        v_ref = np.zeros(3) if v_ref is None else v_ref
        w_ref = np.zeros(3) if w_ref is None else w_ref
        a = np.zeros(7)
        a[:3] = (v_ref * self._dt + self._kp * (p_ref - eef_pos)) / self._pos_scale
        rot_err = (Rotation.from_quat(q_ref) * Rotation.from_quat(eef_quat).inv()).as_rotvec()
        a[3:6] = (w_ref * self._dt + self._kp * rot_err) / self._rot_scale
        return a

    def _start_seg(self, eef_pos, eef_quat, p1, q1, v_lin, w_ang=np.radians(55)):
        q0 = self._seg_q_end if getattr(self, "_seg_q_end", None) is not None else eef_quat
        p0 = self._seg_p_end if getattr(self, "_seg_p_end", None) is not None else eef_pos
        # Start from the previous reference (continuity) unless we drifted far.
        if np.linalg.norm(p0 - eef_pos) > 0.03:
            p0 = eef_pos
        self._seg = _Segment(p0, q0, p1, q1, v_lin * self._style["v"], w_ang * self._style["v"])
        self._seg_p_end, self._seg_q_end = np.asarray(p1, float), np.asarray(q1, float)

    def _follow(self, eef_pos, eef_quat, settle_tol=0.006, settle_cap=0.6):
        """Advance the active segment; return (action, done)."""
        p, q, v, w = self._seg.ref()
        a = self._track(eef_pos, eef_quat, p, q, v, w)
        self._seg.t += self._dt
        done = False
        if self._seg.finished:
            pe = np.linalg.norm(self._seg.p1 - eef_pos)
            re = (Rotation.from_quat(self._seg_q_end) * Rotation.from_quat(eef_quat).inv()).magnitude()
            over = self._seg.t - self._seg.T
            done = (pe < settle_tol and re < np.radians(4)) or over > settle_cap
        return a, done

    # ── planning ───────────────────────────────────────────────────
    def _plan_push(self, obj_pos, obj_quat):
        st = self._style
        # Planned in edge coordinates (u along the edge, w outward); for the
        # +Y edge these are exactly world (x, y).
        c_e = self._to_edge(obj_pos[:2])
        u_t = float(np.clip(c_e[0], *self._u_window)) + st["x_jit"]
        ext_n = self._support(obj_quat, self._n)
        overhang = min(st["overhang"], ext_n - CENTER_MARGIN)
        w_t = TABLE_EDGE_Y + overhang - ext_n
        c_t_e = np.array([u_t, w_t])
        d_e = c_t_e - c_e
        if d_e[1] < 0.03:        # nearly there already: just nudge outward
            d_e = np.array([0.0, 1.0])
        d_e = d_e / np.linalg.norm(d_e)
        # Keep the push mostly outward (a pure-sideways push is unstable).
        if d_e[1] < 0.6:
            d_e = np.array([d_e[0], 0.6]); d_e /= np.linalg.norm(d_e)
        d = self._to_world(d_e)
        ext_d = self._support(obj_quat, d)
        start_xy = obj_pos[:2] - d * (ext_d + PUSH_GAP)
        self._push = dict(c_t_e=c_t_e, d=d, overhang=overhang, start_xy=start_xy, best=np.inf, stall=0.0)

    def _plan_grasp(self, obj_pos, obj_quat):
        st = self._style
        tip = self._tip_n(obj_pos, obj_quat)
        w_g = max(TABLE_EDGE_Y + FINGER_EDGE_CLEAR, tip - st["engage"])
        pts = self._footprint(obj_pos, obj_quat) @ self._R     # -> edge coords (u, w)
        inside = pts[pts[:, 1] > w_g + 0.005]
        u_g = float(inside[:, 0].mean()) if len(inside) else float(self._to_edge(obj_pos[:2])[0])
        z_g = float(obj_pos[2])
        w_pre = max(tip, TABLE_EDGE_Y) + PREGRASP_BACKOFF
        xy = self._to_world([u_g, w_g])
        xy_pre = self._to_world([u_g, w_pre])
        self._grasp = dict(p=np.array([xy[0], xy[1], z_g]),
                           p_pre=np.array([xy_pre[0], xy_pre[1], z_g]),
                           z=z_g, tip=tip)

    # ── Human-like command imperfection (recorded) ─────────────────
    def _ou(self, x, tau, sigma, rng, size):
        rho = np.exp(-self._dt / tau)
        return rho * x + np.sqrt(1 - rho**2) * rng.normal(0.0, sigma, size)

    def _humanize(self, a: np.ndarray) -> np.ndarray:
        """Add operator-like speed variation, wander and tremor to arm dims."""
        k = self.human_noise * self._style["hn"]
        if k <= 0:
            return a
        r = self._rng
        fine = self.phase in self._FINE_PHASES
        dim_w = np.array([1, 1, 1, 0.5, 0.5, 0.5])       # less on rotation
        # 1) speed variation: multiplicative, ~±15 %, 0.6 s timescale
        self._hn_gain = self._ou(self._hn_gain, 0.6, 0.15 * k, r, None)
        # 2) path wander: additive, 0.4 s timescale, scaled down in fine phases
        self._hn_wander = self._ou(self._hn_wander, 0.4, 0.05 * k, r, 6)
        # 3) tremor: fast (~5–10 Hz band), small
        self._hn_tremor = self._ou(self._hn_tremor, 0.025, 0.02 * k, r, 6)
        wander = self._hn_wander * (0.35 if fine else 1.0)
        tremor = self._hn_tremor * (0.7 if fine else 1.0)
        out = a.copy()
        out[:6] = a[:6] * (1.0 + float(np.clip(self._hn_gain, -0.4, 0.4))) \
            + dim_w * (wander + tremor)
        out[:6] = np.clip(out[:6], -1.0, 1.0)
        return out

    # ── DART-style execution noise ─────────────────────────────────
    def perturb(self, action: np.ndarray) -> np.ndarray:
        if self.dart_sigma <= 0:
            return action
        rho = 0.95
        self._noise = rho * self._noise + np.sqrt(1 - rho**2) * self._rng.normal(0, self.dart_sigma, 6)
        out = action.copy()
        out[:6] = np.clip(out[:6] + self._noise, -1.0, 1.0)
        return out

    # ── main step ──────────────────────────────────────────────────
    def act(self, obs: dict, env=None) -> np.ndarray:
        eef = np.asarray(obs["robot0_eef_pos"], float)
        eq = np.asarray(obs.get("robot0_eef_quat", QUAT_DOWN), float)
        op = np.asarray(obs["object_pos"], float)
        oq = np.asarray(obs["object_quat"], float)
        gq = np.asarray(obs.get("robot0_gripper_qpos", [0.04, -0.04]), float)
        opening = float(gq[0] - gq[1])

        a = np.zeros(7)

        if self.phase == "start":
            if env is None:
                raise ValueError("EdgeSlideTeleopPolicy.act needs env= (plate dimensions)")
            self._half = self._read_plate_size(env)
            if self.edge_mode == "closest":
                self._choose_edge(op, oq)
            else:
                self._set_edge(self.edge_mode)
            seed = self._seed
            if seed is None:   # deterministic per scene
                seed = int(abs(op[0] * 1e7) + abs(op[1] * 1e5) + abs(oq[2] * 1e3)) % (2**31)
            self._rng = np.random.default_rng(seed)
            r = self._rng
            self._style = dict(
                v=r.uniform(0.8, 1.25),          # overall speed of this "operator"
                z_travel=TABLE_Z + r.uniform(0.10, 0.13),
                overhang=r.uniform(0.032, 0.045),
                engage=r.uniform(0.026, 0.034),
                lift=r.uniform(0.20, 0.25),
                x_jit=r.uniform(-0.01, 0.01),
                hn=r.uniform(0.7, 1.3),          # this operator's steadiness
            )
            self._seg_p_end = self._seg_q_end = None
            self._gripper = 1.0
            self._advance("approach_push")

        if self.phase == "approach_push":
            if self._seg is None:
                self._plan_push(op, oq)
                sx, sy = self._push["start_xy"]
                self._start_seg(eef, eq, [sx, sy, PUSH_Z + 0.06], QUAT_DOWN, v_lin=0.16)
            a, done = self._follow(eef, eq, settle_tol=0.01)
            if done:
                self._advance("descend_push")

        elif self.phase == "descend_push":
            if self._seg is None:
                sx, sy = self._push["start_xy"]
                self._start_seg(eef, eq, [sx, sy, PUSH_Z], QUAT_DOWN, v_lin=0.08)
            a, done = self._follow(eef, eq)
            if done:
                self._advance("push")

        elif self.phase == "push":
            pu = self._push
            op_e = self._to_edge(op[:2])
            ext_n = self._support(oq, self._n)
            tip = op_e[1] + ext_n
            tip_goal = TABLE_EDGE_Y + min(pu["overhang"], ext_n - CENTER_MARGIN)
            remaining = max(tip_goal - tip, 0.0)
            center_room = max(TABLE_EDGE_Y - CENTER_MARGIN - op_e[1], 0.0)
            remaining = min(remaining, center_room)
            # Re-aim at the goal while far away; freeze the direction for the
            # last few cm so the tangential correction can't swing the push sideways.
            goal_e = np.array([pu["c_t_e"][0], tip_goal - ext_n])
            to_goal = goal_e - op_e
            if remaining > 0.03:
                d_e = to_goal / (np.linalg.norm(to_goal) + 1e-9)
                if d_e[1] < 0.6:
                    d_e = np.array([d_e[0], 0.6]); d_e /= np.linalg.norm(d_e)
                pu["d_now"] = self._to_world(d_e)
            d = pu.get("d_now", pu["d"])
            # Pusher should sit on the line through the plate centre, behind it.
            behind = op[:2] - d * (self._support(oq, d) - 0.004)
            lat = behind - eef[:2]
            lat -= (lat @ d) * d                        # perpendicular part only
            lat = np.clip(lat, -0.01, 0.01)
            ramp = min(1.0, self._t_phase / 0.4)
            v = min(0.06 * self._style["v"], 1.5 * remaining + 0.015) * ramp
            v_xy = v * d + 1.5 * lat
            # Virtual "hand" point that advances at v; the arm is pulled toward
            # it, so contact force builds up if friction holds the pusher back.
            if "ref" not in pu:
                pu["ref"] = eef[:2].copy()
            pu["ref"] = pu["ref"] + v_xy * self._dt
            lag = pu["ref"] - eef[:2]
            if np.linalg.norm(lag) > 0.015:
                pu["ref"] = eef[:2] + lag / np.linalg.norm(lag) * 0.015
            p_ref = np.r_[pu["ref"], PUSH_Z]
            a = self._track(eef, eq, p_ref, QUAT_DOWN, np.r_[v_xy, 0.0])
            a[:2] += self._kpush * (p_ref[:2] - eef[:2]) / self._pos_scale   # extra stiffness along the push

            if remaining <= 0.002:
                self._seg_p_end = self._seg_q_end = None
                self._advance("clear")
            else:
                # Stall / slipped-past detection → back off and re-plan the push.
                if remaining < pu["best"] - 0.001:
                    pu["best"], pu["stall"] = remaining, 0.0
                else:
                    pu["stall"] += self._dt
                passed = self._n @ eef[:2] > self._n @ op[:2] + 0.01
                if (pu["stall"] > 1.5 or passed or self._t_phase > 8.0):
                    self._recover("push stalled")

        elif self.phase == "clear":
            # Small hop up and back from the plate, then on to the grasp side.
            if self._seg is None:
                d = np.r_[self._push["d"], 0.0]
                p1 = eef - 0.02 * d
                p1[2] = self._style["z_travel"]
                self._start_seg(eef, eq, p1, QUAT_DOWN, v_lin=0.12)
            a, done = self._follow(eef, eq, settle_tol=0.012, settle_cap=0.3)
            if done:
                self._advance("to_pregrasp")

        elif self.phase == "to_pregrasp":
            if self._seg is None:
                self._plan_grasp(op, oq)
                g = self._grasp
                self._start_seg(eef, eq, [g["p_pre"][0], g["p_pre"][1], self._style["z_travel"]],
                                self._q_grasp, v_lin=0.16)
            self._gripper = -1.0
            a, done = self._follow(eef, eq, settle_tol=0.012, settle_cap=0.5)
            if done:
                self._advance("descend_pregrasp")

        elif self.phase == "descend_pregrasp":
            if self._seg is None:
                g = self._grasp
                self._start_seg(eef, eq, g["p_pre"], self._q_grasp, v_lin=0.10)
            a, done = self._follow(eef, eq, settle_tol=0.008)
            if done:
                self._advance("approach_grasp")

        elif self.phase == "approach_grasp":
            if self._seg is None:
                self._plan_grasp(op, oq)       # refresh in case the plate moved
                g = self._grasp
                self._start_seg(eef, eq, g["p"], self._q_grasp, v_lin=0.05)
            a, done = self._follow(eef, eq, settle_tol=0.006, settle_cap=0.4)
            if done:
                self._grip_hist = []
                self._advance("close")

        elif self.phase == "close":
            g = self._grasp
            a = self._track(eef, eq, g["p"], self._q_grasp)
            self._gripper = 1.0
            self._grip_hist.append(opening)
            h = self._grip_hist
            w = self._settle_win
            settled = len(h) * self._dt > 0.2 and (max(h[-w:]) - min(h[-w:])) < 1e-4
            if settled or len(h) * self._dt > 1.0:
                if opening < 0.02:                 # fingers met: nothing in between
                    self._recover("missed grasp")
                else:
                    self._lift_ref = (eef.copy(), op.copy())
                    self._advance("lift")

        elif self.phase == "lift":
            if self._seg is None:
                g = self._grasp
                p1 = g["p"] + np.r_[0.05 * self._n, self._style["lift"]]
                self._start_seg(eef, eq, p1, self._q_grasp, v_lin=0.09)
            self._gripper = 1.0
            a, done = self._follow(eef, eq, settle_tol=0.01, settle_cap=1.0)
            e0, o0 = self._lift_ref
            de, do = eef[2] - e0[2], op[2] - o0[2]
            if de > 0.04 and do < 0.5 * de:
                self._recover("plate slipped")
            elif done:
                self._advance("hold")

        elif self.phase == "back_off":
            # Open, retreat to the pre-grasp stand-off, then decide.
            if self._seg is None:
                p1 = eef.copy()
                w_eef = self._n @ eef[:2]
                w_tgt = max(w_eef, max(self._tip_n(op, oq), TABLE_EDGE_Y) + PREGRASP_BACKOFF)
                p1[:2] = eef[:2] + (w_tgt - w_eef) * self._n
                p1[2] = max(eef[2], op[2])
                self._start_seg(eef, eq, p1, self._q_grasp, v_lin=0.08)
            self._gripper = -1.0
            a, done = self._follow(eef, eq, settle_tol=0.012, settle_cap=0.4)
            if done:
                overhang = self._tip_n(op, oq) - TABLE_EDGE_Y
                fell = op[2] < TABLE_Z - 0.03
                if fell:
                    self._advance("hold")
                elif overhang >= MIN_OVERHANG:
                    self._style["engage"] = min(self._style["engage"] + 0.004, 0.036)
                    self._seg_p_end = self._seg_q_end = None
                    self._advance("descend_pregrasp")
                    self._plan_grasp(op, oq)
                else:
                    self._advance("return_to_push")

        elif self.phase == "return_to_push":
            if self._seg is None:
                p1 = eef.copy(); p1[2] = self._style["z_travel"]
                self._start_seg(eef, eq, p1, QUAT_DOWN, v_lin=0.12)
            self._gripper = 1.0
            a, done = self._follow(eef, eq, settle_tol=0.015, settle_cap=0.3)
            if done:
                self._advance("approach_push")

        elif self.phase == "hold":
            if self._seg is None:
                self._hold_p, self._hold_q = eef.copy(), eq.copy()
                self._seg = True
            a = self._track(eef, eq, self._hold_p, self._hold_q)

        # ── shape the command like a hand on a 6-DoF mouse ─────────
        a[:6] = np.clip(a[:6], -A_MAX, A_MAX)
        a[:6] = self._prev_action[:6] + np.clip(a[:6] - self._prev_action[:6], -self._slew, self._slew)
        a[6] = self._gripper
        self._prev_action = a.copy()          # smooth "intent", before noise
        a = self._humanize(a)
        self._t_phase += self._dt
        return a

    def _recover(self, why: str):
        if self.verbose:
            print(f"  [recover] {why} (phase={self.phase}, recoveries={self._recoveries})", flush=True)
        self._recoveries += 1
        self._seg_p_end = self._seg_q_end = None
        if self._recoveries > MAX_RECOVERIES:
            self._advance("hold")
        elif self.phase == "push":
            self._advance("return_to_push")
        else:
            self._advance("back_off")


# ── Standalone runner ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    import imbench as ib

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 14])
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--dart-sigma", type=float, default=0.0)
    ap.add_argument("--human-noise", type=float, default=0.0)
    ap.add_argument("--control-freq", type=int, default=100)
    ap.add_argument("--edge-mode", default="+y", help="+y | -y | +x | closest")
    ap.add_argument("--allowed-edges", nargs="+", default=["+y", "-y"])
    args = ap.parse_args()

    env = ib.make("ib-p1-edge-slide-v1",
                  obs_mode="combo" if args.render else "proprio",
                  action_mode="cartesian_delta",
                  render_mode="human" if args.render else None,
                  config_overrides={"control_freq": args.control_freq})
    policy = EdgeSlideTeleopPolicy(dart_sigma=args.dart_sigma, human_noise=args.human_noise,
                                   control_freq=args.control_freq, edge_mode=args.edge_mode,
                                   allowed_edges=args.allowed_edges)
    for seed in args.seeds:
        obs, info = env.reset(seed=seed)
        policy.reset()
        obs, *_, info = env.step(np.zeros(7))
        print(f"\n=== seed {seed}  plate={np.round(obs['object_pos'], 3)}")
        done, step = False, 0
        while not done:
            act = policy.act(obs, env=env)
            obs, r, term, trunc, info = env.step(policy.perturb(act))
            if args.render:
                env.render()
            done = term or trunc
            step += 1
        print(f"seed {seed}: edge={policy.edge} score={info.get('task_score', 0):.1f} "
              f"steps={step} phase={policy.phase}")
    env.close()
