"""Motor primitives on top of the edge-slide OSC_POSE delta controller.

Conventions (verified in recon, see README "Stage 3"):
- Positions: metres, WORLD frame. The robot base frame has identity orientation (base at
  (-0.41, 0, 0.912)), so base-frame deltas == world-frame deltas.
- Orientation: roll/pitch/yaw in DEGREES, EXTRINSIC XYZ about the fixed world axes
  (scipy ``Rotation.from_euler("xyz", ...)``, lowercase = extrinsic). Observation readback and
  command schema use the same convention.
- robosuite eef quaternion is xyzw (scipy order).
- Action: [dx, dy, dz, dax, day, daz, grip] in [-1, 1]. OSC re-targets each tick to
  achieved_pose + a * output_max, with output_max = 0.05 m / 0.5 rad, rotation deltas as an
  axis-angle vector about WORLD axes. Gripper: -1 open, +1 closed.
- Control at 100 Hz (1 tick = 1 env.step = 5 MuJoCo substeps of 2 ms).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

RPY_CONVENTION = "xyz"                 # extrinsic X-Y-Z about world axes, degrees
POS_OUTPUT_MAX = 0.05                  # m per tick at |a|=1 (controller output_max)
ROT_OUTPUT_MAX = 0.5                   # rad per tick at |a|=1
CONTROL_HZ = 100

# MOVE_EEF tolerances / gains (tuned in tests/test_primitives.py)
POS_TOL = 0.005                        # m
ROT_TOL_DEG = 5.0
KP_POS = 1.0                           # a = Kp*e/output_max -> OSC goal = achieved + Kp*e
KD_POS = 0.08                          # s, damping on measured eef velocity (gain sweep in report)
KD_ROT = 0.1                           # s, damping on measured angular velocity
KP_ROT = 1.0
ROT_CAP = 0.3                          # |a_rot| cap: saturated rotation (1.0) dragged position off in roll/pitch
VEL_TOL = 0.02                         # m/s: must be nearly still to count as converged
ANGVEL_TOL_DEG = 5.0                   # deg/s
MAX_SPEED_FAR = 1.0                    # |a| cap far from target
SLOW_RADIUS = 0.02                     # m: below this, |a_pos| is capped proportionally
SETTLE_TICKS = 3                       # consecutive in-tolerance ticks needed to declare convergence

# gripper
GRIP_SETTLE_EPS = 2e-4                 # m change in width per tick considered "stopped"
GRIP_MAX_TICKS = 120
EMPTY_CLOSED_WIDTH = 0.0095            # m, measured: fingers fully closed on nothing
OBJECT_WIDTH_MARGIN = 0.003            # width above empty-closed => something between fingers

CONTACT_FORCE_N = 10.0                 # |F_world - F_baseline| above this => contact
ATOMIC_MAX_TICKS = 200
DEFAULT_MAX_STEPS = 400


def quat_to_rpy_deg(q_xyzw) -> np.ndarray:
    return R.from_quat(q_xyzw).as_euler(RPY_CONVENTION, degrees=True)


def rpy_deg_to_rot(rpy) -> R:
    return R.from_euler(RPY_CONVENTION, rpy, degrees=True)


def rot_error_vec(r_target: R, r_current: R) -> np.ndarray:
    """Axis-angle (world frame, rad) taking r_current to r_target: R_t * R_c^T."""
    return (r_target * r_current.inv()).as_rotvec()


class GripperRegister:
    """Remembers the last OPEN/CLOSE command (per robot) so motion primitives never change it."""

    def __init__(self, value: float = -1.0):
        self.value = float(value)

    def set(self, v: float):
        self.value = float(np.clip(v, -1.0, 1.0))


@dataclass
class EpisodeState:
    ticks: int = 0
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    success: bool = False
    done_called: bool = False
    frames: list = field(default_factory=list)


class PrimitiveExecutor:
    """Executes primitives on an IMBench task. Holds the latest obs and the gripper register."""

    def __init__(self, task, obs: dict, record_cam: str | tuple | None = "exo_right_cam", record_every: int = 4,
                 tick_cap: int = 10_000):
        self.task = task
        self.obs = obs
        self.reg = GripperRegister(-1.0)
        self.st = EpisodeState()
        self.record_cam = record_cam
        self.record_every = record_every
        self.tick_cap = tick_cap
        self._prev_pos = np.array(obs["robot0_eef_pos"], float)
        self._prev_rot = R.from_quat(obs["robot0_eef_quat"])
        if record_cam:
            self.st.frames.append(self._frame())

    # ── state readback ─────────────────────────────────────────────────
    @property
    def pos(self) -> np.ndarray:
        return np.array(self.obs["robot0_eef_pos"], float)

    @property
    def rot(self) -> R:
        return R.from_quat(self.obs["robot0_eef_quat"])

    @property
    def rpy(self) -> np.ndarray:
        return quat_to_rpy_deg(self.obs["robot0_eef_quat"])

    @property
    def width(self) -> float:
        g = self.obs["robot0_gripper_qpos"]
        return float(abs(g[0]) + abs(g[1]))

    def force_world(self) -> np.ndarray:
        return np.array(self.obs.get("force_torque_world", np.zeros(3)), float)

    def tactile_max(self) -> float:
        return float(max(np.max(self.obs.get("robot0_tactile_left", 0)),
                         np.max(self.obs.get("robot0_tactile_right", 0))))

    @property
    def stopped(self) -> bool:
        return self.st.terminated or self.st.success or self.st.ticks >= self.tick_cap

    # ── one control tick ───────────────────────────────────────────────
    def tick(self, action: np.ndarray) -> None:
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        self._prev_pos = self.pos
        self._prev_rot = self.rot
        self.obs, r, term, trunc, info = self.task.step(a)
        self.st.ticks += 1
        self.st.reward = float(r)
        self.st.terminated = bool(term)
        self.st.truncated = bool(trunc)
        if info.get("task_score", r) >= 1.0:
            self.st.success = True
        if self.record_cam and self.st.ticks % self.record_every == 0:
            self.st.frames.append(self._frame())

    def _frame(self) -> np.ndarray:
        cams = (self.record_cam,) if isinstance(self.record_cam, str) else tuple(self.record_cam)
        return np.concatenate([self.obs[f"rgb_{c}"] for c in cams], axis=1).copy()

    def velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """(linear m/s, angular rad/s world) from the last tick."""
        v = (self.pos - self._prev_pos) * CONTROL_HZ
        w = (self.rot * self._prev_rot.inv()).as_rotvec() * CONTROL_HZ
        return v, w

    def servo_action(self, p_t: np.ndarray, r_t: R, speed_cap: bool = True) -> np.ndarray:
        """One tick of the pose servo toward (p_t, r_t), with damping and speed saturation."""
        e_p = p_t - self.pos
        e_r = rot_error_vec(r_t, self.rot)
        v, w = self.velocity()
        a_pos = (KP_POS * e_p - KD_POS * v) / POS_OUTPUT_MAX
        if speed_cap:
            dist = np.linalg.norm(e_p)
            cap = MAX_SPEED_FAR if dist > SLOW_RADIUS else max(0.15, MAX_SPEED_FAR * dist / SLOW_RADIUS)
            n = np.max(np.abs(a_pos))
            if n > cap:
                a_pos = a_pos * cap / n
        a_rot = (KP_ROT * e_r - KD_ROT * w) / ROT_OUTPUT_MAX
        n = np.max(np.abs(a_rot))
        if n > ROT_CAP:
            a_rot = a_rot * ROT_CAP / n
        return np.concatenate([np.clip(a_pos, -1, 1), a_rot, [self.reg.value]])

    # ── primitives ─────────────────────────────────────────────────────
    def move_eef(self, position_m, orientation_rpy_deg=None, max_steps: int = DEFAULT_MAX_STEPS) -> dict:
        p_t = np.asarray(position_m, float)
        r_t = self.rot if orientation_rpy_deg is None else rpy_deg_to_rot(orientation_rpy_deg)
        f0 = self.force_world()
        peak_f, contact, steps, inside = 0.0, False, 0, 0
        converged = False
        while steps < max_steps and not self.stopped:
            e_p = p_t - self.pos
            e_r = rot_error_vec(r_t, self.rot)
            v, w = self.velocity()
            if (np.linalg.norm(e_p) < POS_TOL and np.degrees(np.linalg.norm(e_r)) < ROT_TOL_DEG
                    and np.linalg.norm(v) < VEL_TOL and np.degrees(np.linalg.norm(w)) < ANGVEL_TOL_DEG):
                inside += 1
                if inside >= SETTLE_TICKS:
                    converged = True
                    break
            else:
                inside = 0
            self.tick(self.servo_action(p_t, r_t))
            steps += 1
            df = np.linalg.norm(self.force_world() - f0)
            peak_f = max(peak_f, df)
            if df > CONTACT_FORCE_N or self.tactile_max() > 0:
                contact = True
        e_p = p_t - self.pos
        e_r = rot_error_vec(r_t, self.rot)
        return {"converged": converged, "steps": steps,
                "final_position_m": _r(self.pos, 4), "final_rpy_deg": _r(self.rpy, 2),
                "position_error_mm": round(float(np.linalg.norm(e_p)) * 1000, 2),
                "orientation_error_deg": round(float(np.degrees(np.linalg.norm(e_r))), 2),
                "peak_force_N": round(peak_f, 2), "contact": contact}

    def _actuate_gripper(self, value: float) -> dict:
        self.reg.set(value)
        hold_p, hold_r = self.pos, self.rot
        last, still, n = self.width, 0, 0
        while n < GRIP_MAX_TICKS and not self.stopped:
            self.tick(self.servo_action(hold_p, hold_r))
            n += 1
            w = self.width
            still = still + 1 if abs(w - last) < GRIP_SETTLE_EPS else 0
            last = w
            if still >= 5 and n >= 10:
                break
        w = self.width
        return {"finger_width_m": round(w, 4), "ticks": n,
                "object_between_fingers": bool(value > 0 and w > EMPTY_CLOSED_WIDTH + OBJECT_WIDTH_MARGIN),
                "tactile_max": round(self.tactile_max(), 4)}

    def close_gripper(self) -> dict:
        return self._actuate_gripper(1.0)

    def open_gripper(self) -> dict:
        return self._actuate_gripper(-1.0)

    def lift(self, dz_m: float, max_steps: int = DEFAULT_MAX_STEPS) -> dict:
        p = self.pos
        return self.move_eef([p[0], p[1], p[2] + float(dz_m)], self.rpy, max_steps)

    def descend(self, target_z_m: float, max_steps: int = DEFAULT_MAX_STEPS) -> dict:
        p = self.pos
        return self.move_eef([p[0], p[1], float(target_z_m)], self.rpy, max_steps)

    def atomic_action(self, action7, n_ticks: int) -> dict:
        a = np.clip(np.asarray(action7, float), -1, 1)
        self.reg.set(a[6])
        p0 = self.pos
        n = int(np.clip(n_ticks, 0, ATOMIC_MAX_TICKS))
        k = 0
        while k < n and not self.stopped:
            self.tick(a)
            k += 1
        return {"ticks": k, "displacement_m": _r(self.pos - p0, 4), "final_position_m": _r(self.pos, 4),
                "final_rpy_deg": _r(self.rpy, 2)}

    def hold(self, n_ticks: int) -> dict:
        p0, r0 = self.pos, self.rot
        n = int(np.clip(n_ticks, 0, ATOMIC_MAX_TICKS))
        k = 0
        while k < n and not self.stopped:
            self.tick(self.servo_action(p0, r0))
            k += 1
        return {"ticks": k, "drift_mm": round(float(np.linalg.norm(self.pos - p0)) * 1000, 2)}

    def done(self) -> dict:
        self.st.done_called = True
        return {"done": True}

    # ── dispatch by tool name ──────────────────────────────────────────
    def run(self, name: str, params: dict) -> dict:
        n = name.upper()
        if n == "MOVE_EEF":
            return self.move_eef(params["position_m"], params.get("orientation_rpy_deg"),
                                 int(params.get("max_steps") or DEFAULT_MAX_STEPS))
        if n == "CLOSE_GRIPPER":
            return self.close_gripper()
        if n == "OPEN_GRIPPER":
            return self.open_gripper()
        if n == "LIFT":
            return self.lift(params["dz_m"], int(params.get("max_steps") or DEFAULT_MAX_STEPS))
        if n == "DESCEND":
            return self.descend(params["target_z_m"], int(params.get("max_steps") or DEFAULT_MAX_STEPS))
        if n == "ATOMIC_ACTION":
            return self.atomic_action(params["action7"], int(params["n_ticks"]))
        if n == "HOLD":
            return self.hold(int(params.get("n_ticks", 20)))
        if n == "DONE":
            return self.done()
        raise ValueError(f"unknown primitive {name!r}")


def _r(a, n):
    return [round(float(x), n) for x in a]
