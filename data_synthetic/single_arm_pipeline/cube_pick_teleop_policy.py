"""
cube_pick_teleop_policy.py
==========================
Teleoperator-style scripted policy for ``ib-p1-cube-pick-v1``: top-down grasp
across the box's short horizontal side, then lift.

Shares the motion machinery of :class:`EdgeSlideTeleopPolicy` (min-jerk
segments, feed-forward + P tracking, slew limiting, per-episode style and
human-like noise), so the edge-slide and cube datasets have one demonstrator
"style".

Phases: start, approach, descend, close, lift, hold (+ back_off on a miss/slip).

    python data_synthetic/single_arm_pipeline/cube_pick_teleop_policy.py --seeds 0 1 2
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

if __name__ == "__main__":   # allow running as a script from anywhere
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data_synthetic.single_arm_pipeline.edge_slide_teleop_policy import (
    A_MAX, QUAT_DOWN, TABLE_Z, EdgeSlideTeleopPolicy,
)

MAX_RECOVERIES = 3


class CubePickTeleopPolicy(EdgeSlideTeleopPolicy):

    _FINE_PHASES = {"descend", "close", "lift"}

    def reset(self, seed: int | None = None):
        super().reset(seed)
        self._gripper = -1.0
        self._q_pick = QUAT_DOWN.copy()

    # ── planning ───────────────────────────────────────────────────
    def _plan_pick(self, op, oq):
        """Top-down orientation closing across the box's short side, and grasp height."""
        ax, ay = self._axes_xy(oq)
        short = ay if self._half[1] <= self._half[0] else ax
        # At QUAT_DOWN the fingers close along world Y; yaw so they close along `short`.
        psi = np.arctan2(short[1], short[0]) - np.pi / 2
        psi = (psi + np.pi / 2) % np.pi - np.pi / 2          # symmetric jaw: keep |psi| <= 90°
        self._q_pick = (Rotation.from_euler("z", psi) * Rotation.from_quat(QUAT_DOWN)).as_quat()
        top = op[2] + self._half[2]
        z_g = max(top - self._style["depth"], TABLE_Z + 0.014)
        self._grasp = dict(p=np.array([op[0], op[1], z_g]), top=top)

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
                raise ValueError("CubePickTeleopPolicy.act needs env= (box dimensions)")
            self._half = self._read_plate_size(env)
            seed = self._seed
            if seed is None:   # deterministic per scene
                seed = int(abs(op[0] * 1e7) + abs(op[1] * 1e5) + abs(oq[2] * 1e3)) % (2**31)
            self._rng = np.random.default_rng(seed)
            r = self._rng
            self._style = dict(
                v=r.uniform(0.8, 1.25),
                hover=r.uniform(0.07, 0.11),        # pre-grasp height above the box top
                depth=r.uniform(0.012, 0.022),      # how far below the top the fingers grip
                lift=r.uniform(0.20, 0.25),
                xy_jit=r.uniform(-0.004, 0.004, 2),
                hn=r.uniform(0.7, 1.3),
            )
            self._seg_p_end = self._seg_q_end = None
            self._gripper = -1.0
            self._advance("approach")

        if self.phase == "approach":
            if self._seg is None:
                self._plan_pick(op, oq)
                g = self._grasp
                p1 = np.r_[op[:2] + self._style["xy_jit"], g["top"] + self._style["hover"]]
                self._start_seg(eef, eq, p1, self._q_pick, v_lin=0.16)
            self._gripper = -1.0
            a, done = self._follow(eef, eq, settle_tol=0.008, settle_cap=0.5)
            if done:
                self._advance("descend")

        elif self.phase == "descend":
            if self._seg is None:
                self._plan_pick(op, oq)      # refresh in case the box was nudged
                p1 = self._grasp["p"].copy()
                p1[:2] += self._style["xy_jit"]
                self._start_seg(eef, eq, p1, self._q_pick, v_lin=0.08)
            a, done = self._follow(eef, eq, settle_tol=0.005, settle_cap=0.5)
            if done:
                self._grip_hist = []
                self._advance("close")

        elif self.phase == "close":
            p = self._seg_p_end
            a = self._track(eef, eq, p, self._q_pick)
            self._gripper = 1.0
            self._grip_hist.append(opening)
            h, w = self._grip_hist, self._settle_win
            settled = len(h) * self._dt > 0.2 and (max(h[-w:]) - min(h[-w:])) < 1e-4
            if settled or len(h) * self._dt > 1.0:
                if opening < 0.015:          # fingers met: missed
                    self._recover("missed grasp")
                else:
                    self._lift_ref = (eef.copy(), op.copy())
                    self._advance("lift")

        elif self.phase == "lift":
            if self._seg is None:
                p1 = eef + np.array([0.0, 0.0, self._style["lift"]])
                self._start_seg(eef, eq, p1, self._q_pick, v_lin=0.09)
            self._gripper = 1.0
            a, done = self._follow(eef, eq, settle_tol=0.01, settle_cap=1.0)
            e0, o0 = self._lift_ref
            de, do = eef[2] - e0[2], op[2] - o0[2]
            if de > 0.04 and do < 0.5 * de:
                self._recover("box slipped")
            elif done:
                self._advance("hold")

        elif self.phase == "back_off":
            if self._seg is None:
                p1 = eef.copy()
                p1[2] = max(eef[2], op[2] + self._half[2] + 0.08)
                self._start_seg(eef, eq, p1, self._q_pick, v_lin=0.10)
            self._gripper = -1.0
            a, done = self._follow(eef, eq, settle_tol=0.012, settle_cap=0.4)
            if done:
                if op[2] < TABLE_Z - 0.03:
                    self._advance("hold")
                else:
                    self._seg_p_end = self._seg_q_end = None
                    self._advance("approach")

        elif self.phase == "hold":
            if self._seg is None:
                self._hold_p, self._hold_q = eef.copy(), eq.copy()
                self._seg = True
            a = self._track(eef, eq, self._hold_p, self._hold_q)

        a[:6] = np.clip(a[:6], -A_MAX, A_MAX)
        a[:6] = self._prev_action[:6] + np.clip(a[:6] - self._prev_action[:6], -self._slew, self._slew)
        a[6] = self._gripper
        self._prev_action = a.copy()
        a = self._humanize(a)
        self._t_phase += self._dt
        return a

    def _recover(self, why: str):
        if self.verbose:
            print(f"  [recover] {why} (phase={self.phase}, recoveries={self._recoveries})", flush=True)
        self._recoveries += 1
        self._seg_p_end = self._seg_q_end = None
        self._advance("hold" if self._recoveries > MAX_RECOVERIES else "back_off")


if __name__ == "__main__":
    import argparse
    import imbench as ib

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--human-noise", type=float, default=1.0)
    ap.add_argument("--control-freq", type=int, default=20)
    args = ap.parse_args()
    env = ib.make("ib-p1-cube-pick-v1", obs_mode="combo" if args.render else "proprio",
                  action_mode="cartesian_delta", render_mode="human" if args.render else None,
                  max_episode_steps=1000, config_overrides={"control_freq": args.control_freq})
    policy = CubePickTeleopPolicy(human_noise=args.human_noise, control_freq=args.control_freq)
    for seed in args.seeds:
        obs, info = env.reset(seed=seed)
        policy.reset()
        done, step = False, 0
        while not done:
            obs, r, term, trunc, info = env.step(np.clip(policy.act(obs, env=env), -1, 1))
            if args.render:
                env.render()
            done = term or trunc
            step += 1
        print(f"seed {seed}: score={info.get('task_score', 0):.1f} steps={step} phase={policy.phase}")
    env.close()
