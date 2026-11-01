"""Sim tests of the 8 primitives on edge-slide (no model). Writes runs/stage3_primitive_report.md.

    python -m imbench.vlm_eval.stage3.primitive_report --seeds 0 1 2

tests/test_primitives.py runs the same checks and asserts on them.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from imbench.vlm_eval.stage3 import primitives as P

TASK_ID = "ib-p1-edge-slide-v1"
TABLE_Z = 0.80          # recon: table top surface z
CAMS = ("exo_right_cam", "gripper_cam")


@dataclass
class Row:
    seed: int
    group: str
    name: str
    target: str
    achieved: str
    error: str
    steps: int
    passed: bool
    note: str = ""
    frames: str = ""
    pos_err_mm: float | None = None      # for commanded-vs-achieved stats (converged MOVE_EEF only)
    rot_err_deg: float | None = None


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)

    def add(self, r: Row):
        self.rows.append(r)
        flag = "PASS" if r.passed else "FAIL"
        print(f"[{flag}] s{r.seed} {r.group:11s} {r.name:28s} err {r.error:24s} steps {r.steps:4d} {r.note}",
              flush=True)


def _fmt(v, n=3):
    return "[" + ", ".join(f"{x:.{n}f}" for x in v) + "]"


class Harness:
    def __init__(self, seed: int, out_dir: Path, rep: Report):
        import imbench as ib
        self.seed, self.out, self.rep = seed, out_dir, rep
        self.task = ib.make(TASK_ID, obs_mode="combo", max_episode_steps=100_000)
        obs, _ = self.task.reset(seed=seed)
        self.ex = P.PrimitiveExecutor(self.task, obs, record_cam=None)
        self.ex.hold(30)
        self.home = self.ex.pos.copy()
        self.home_rpy = self.ex.rpy.copy()

    def snap(self) -> list[np.ndarray]:
        return [self.ex.obs[f"rgb_{c}"].copy() for c in CAMS]

    def save_frames(self, name: str, before, after) -> str:
        d = self.out / "frames"
        d.mkdir(parents=True, exist_ok=True)
        tiles = [Image.fromarray(x).resize((256, 192)) for x in before + after]
        sheet = Image.new("RGB", (512, 384), "white")
        for i, t in enumerate(tiles):       # rows: before/after, cols: agentview/gripper
            sheet.paste(t, ((i % 2) * 256, (i // 2) * 192))
        fn = f"s{self.seed}_{name.replace(' ', '_').replace('/', '_')}.jpg"
        sheet.save(d / fn, quality=85)
        return f"frames/{fn}"

    def go_home(self):
        if self.ex.pos[2] < self.home[2] - 0.03:
            self.ex.lift(self.home[2] - self.ex.pos[2])
        self.ex.move_eef(self.home, self.home_rpy, 400)
        self.ex.hold(20)

    def close(self):
        self.task.close()

    # ── position ──────────────────────────────────────────────────────
    def axis_moves(self):
        for ax, axn in enumerate("xyz"):
            for d in (0.05, -0.05, 0.10, -0.10):
                self.go_home()
                p0, r0, b = self.ex.pos.copy(), self.ex.rot, self.snap()
                tgt = p0.copy()
                tgt[ax] += d
                res = self.ex.move_eef(tgt, None, 400)
                disp = self.ex.pos - p0
                sign_ok = np.sign(disp[ax]) == np.sign(d) and abs(disp[ax]) == np.max(np.abs(disp))
                ori_drift = np.degrees((self.ex.rot * r0.inv()).magnitude())
                ok = res["converged"] and res["position_error_mm"] < 5 and sign_ok and ori_drift < 2
                self.rep.add(Row(self.seed, "position", f"axis {axn} {d:+.2f} m", _fmt(tgt), _fmt(self.ex.pos),
                                 f"{res['position_error_mm']:.2f} mm", res["steps"], ok,
                                 f"disp {_fmt(disp)} sign_ok={sign_ok} ori_drift={ori_drift:.2f}deg",
                                 self.save_frames(f"axis_{axn}{d:+.2f}", b, self.snap()),
                                 res["position_error_mm"], res["orientation_error_deg"]))

    def random_targets(self, n=5):
        rng = np.random.RandomState(100 + self.seed)
        for i in range(n):
            self.go_home()
            tgt = np.array([rng.uniform(-0.15, 0.20), rng.uniform(-0.20, 0.20), rng.uniform(0.90, 1.12)])
            b = self.snap()
            res = self.ex.move_eef(tgt, None, 500)
            ok = res["converged"] and res["position_error_mm"] < 5
            self.rep.add(Row(self.seed, "position", f"random target {i}", _fmt(tgt), _fmt(self.ex.pos),
                             f"{res['position_error_mm']:.2f} mm", res["steps"], ok, "",
                             self.save_frames(f"random_{i}", b, self.snap()),
                             res["position_error_mm"], res["orientation_error_deg"]))

    def below_table(self):
        self.go_home()
        plate_y = float(self.ex.obs["object_pos"][1])
        y = -0.20 if plate_y > 0 else 0.20                 # table spot away from the plate
        self.ex.move_eef([self.home[0], y, 0.90], None, 400)
        b = self.snap()
        tgt = np.array([self.home[0], y, TABLE_Z - 0.05])
        res = self.ex.move_eef(tgt, None, 200)
        finite = bool(np.all(np.isfinite(self.ex.pos)))
        ok = (not res["converged"]) and res["contact"] and finite and not self.ex.st.terminated
        self.rep.add(Row(self.seed, "position", "target 5 cm below table", _fmt(tgt), _fmt(self.ex.pos),
                         f"{res['position_error_mm']:.1f} mm", res["steps"], ok,
                         f"converged={res['converged']} contact={res['contact']} peakF={res['peak_force_N']}N "
                         f"finite={finite}", self.save_frames("below_table", b, self.snap())))
        self.ex.lift(0.10)

    # ── orientation ───────────────────────────────────────────────────
    def single_rotations(self):
        for idx, nm, d in [(2, "yaw", 45), (2, "yaw", -45), (2, "yaw", 90), (2, "yaw", -90),
                           (0, "roll", 20), (0, "roll", -20), (1, "pitch", 20), (1, "pitch", -20)]:
            self.go_home()
            p0, r0, b = self.ex.pos.copy(), self.ex.rot, self.snap()
            rpy_t = self.home_rpy.copy()
            rpy_t[idx] += d
            res = self.ex.move_eef(p0, rpy_t, 500)
            pdrift = np.linalg.norm(self.ex.pos - p0) * 1000
            ok = res["converged"] and res["orientation_error_deg"] < 5 and pdrift < 5
            note = f"pos drift {pdrift:.2f} mm"
            if nm == "yaw" and d > 0:
                x0 = r0.apply([1, 0, 0])
                x1 = self.ex.rot.apply([1, 0, 0])
                ccw = np.cross(x0[:2].tolist() + [0], x1[:2].tolist() + [0])[2] > 0
                readback = self.ex.rpy[2] - self.home_rpy[2]
                note += f"; x-axis CCW from above={ccw}; readback dyaw={readback:+.1f}deg"
                ok = ok and ccw and abs(((readback - d) + 180) % 360 - 180) < 5
            self.rep.add(Row(self.seed, "orientation", f"{nm} {d:+d} deg", _fmt(rpy_t, 1), _fmt(self.ex.rpy, 1),
                             f"{res['orientation_error_deg']:.2f} deg", res["steps"], ok, note,
                             self.save_frames(f"{nm}{d:+d}", b, self.snap()),
                             res["position_error_mm"], res["orientation_error_deg"]))

    def combined_and_roundtrip(self):
        self.go_home()
        tgt = self.home + np.array([0.06, -0.08, -0.07])
        rpy_t = self.home_rpy + np.array([0, 10, 35])
        b = self.snap()
        res = self.ex.move_eef(tgt, rpy_t, 600)
        ok = res["converged"] and res["position_error_mm"] < 5 and res["orientation_error_deg"] < 5
        self.rep.add(Row(self.seed, "orientation", "combined pos+ori", _fmt(tgt) + " " + _fmt(rpy_t, 1),
                         _fmt(self.ex.pos) + " " + _fmt(self.ex.rpy, 1),
                         f"{res['position_error_mm']:.2f} mm / {res['orientation_error_deg']:.2f} deg",
                         res["steps"], ok, "", self.save_frames("combined", b, self.snap()),
                         res["position_error_mm"], res["orientation_error_deg"]))
        # round trip: command exactly the read-back pose
        p0, rpy0, r0 = self.ex.pos.copy(), self.ex.rpy.copy(), self.ex.rot
        res = self.ex.move_eef(p0, rpy0, 200)
        dp = np.linalg.norm(self.ex.pos - p0) * 1000
        dr = np.degrees((self.ex.rot * r0.inv()).magnitude())
        ok = res["converged"] and dp < 2 and dr < 1
        self.rep.add(Row(self.seed, "orientation", "round trip (readback->command)", _fmt(p0) + " " + _fmt(rpy0, 1),
                         _fmt(self.ex.pos) + " " + _fmt(self.ex.rpy, 1), f"{dp:.2f} mm / {dr:.2f} deg",
                         res["steps"], ok, "expect ~zero motion"))

    # ── others ───────────────────────────────────────────────────────
    def gripper(self):
        self.go_home()
        b = self.snap()
        rc = self.ex.close_gripper()
        ok = rc["finger_width_m"] < 0.015 and not rc["object_between_fingers"]
        self.rep.add(Row(self.seed, "gripper", "CLOSE (empty)", "width<0.015", f"{rc['finger_width_m']:.4f}",
                         "-", rc["ticks"], ok, f"object_between={rc['object_between_fingers']}"))
        w_closed = self.ex.width
        res = self.ex.move_eef(self.home + np.array([0.08, 0.05, -0.05]), None, 400)
        ok = abs(self.ex.width - w_closed) < 0.002 and self.ex.reg.value == 1.0
        self.rep.add(Row(self.seed, "gripper", "MOVE_EEF keeps CLOSED", f"{w_closed:.4f}", f"{self.ex.width:.4f}",
                         f"{abs(self.ex.width - w_closed) * 1000:.2f} mm", res["steps"], ok,
                         f"register={self.ex.reg.value}", self.save_frames("closed_move", b, self.snap())))
        ro = self.ex.open_gripper()
        ok = ro["finger_width_m"] > 0.07
        self.rep.add(Row(self.seed, "gripper", "OPEN", "width>0.07", f"{ro['finger_width_m']:.4f}", "-",
                         ro["ticks"], ok))

    def lift_descend_hold(self):
        self.go_home()
        p0 = self.ex.pos.copy()
        res = self.ex.descend(0.88)
        xy = np.linalg.norm(self.ex.pos[:2] - p0[:2]) * 1000
        ok = res["converged"] and xy < 5 and abs(self.ex.pos[2] - 0.88) < 0.005
        self.rep.add(Row(self.seed, "lift/desc", "DESCEND to z=0.88", "z=0.880", f"z={self.ex.pos[2]:.3f}",
                         f"xy drift {xy:.2f} mm", res["steps"], ok, "", None,
                         res["position_error_mm"], res["orientation_error_deg"]))
        p1 = self.ex.pos.copy()
        res = self.ex.lift(0.12)
        xy = np.linalg.norm(self.ex.pos[:2] - p1[:2]) * 1000
        ok = res["converged"] and xy < 5 and abs(self.ex.pos[2] - p1[2] - 0.12) < 0.005
        self.rep.add(Row(self.seed, "lift/desc", "LIFT +0.12 m", f"z={p1[2] + 0.12:.3f}", f"z={self.ex.pos[2]:.3f}",
                         f"xy drift {xy:.2f} mm", res["steps"], ok, "", None,
                         res["position_error_mm"], res["orientation_error_deg"]))
        rh = self.ex.hold(50)
        self.rep.add(Row(self.seed, "lift/desc", "HOLD 50 ticks", "drift<2mm", f"{rh['drift_mm']:.2f} mm",
                         f"{rh['drift_mm']:.2f} mm", rh["ticks"], rh["drift_mm"] < 2))

    def atomic(self):
        for i, nm in enumerate(["dx", "dy", "dz", "droll", "dpitch", "dyaw"]):
            for sgn in (1, -1):
                self.go_home()
                p0, r0 = self.ex.pos.copy(), self.ex.rot
                a = np.zeros(7)
                a[i] = 0.5 * sgn
                a[6] = -1
                res = self.ex.atomic_action(a, 20)
                self.ex.hold(10)
                if i < 3:
                    d = self.ex.pos - p0
                    ok = np.sign(d[i]) == sgn and abs(d[i]) == np.max(np.abs(d))
                    got = f"disp {_fmt(d)} m"
                else:
                    d = (self.ex.rot * r0.inv()).as_rotvec()
                    ok = np.sign(d[i - 3]) == sgn and abs(d[i - 3]) == np.max(np.abs(d))
                    got = f"world rotvec {_fmt(np.degrees(d), 2)} deg"
                self.rep.add(Row(self.seed, "atomic", f"ATOMIC {nm} {sgn:+d}x0.5 x20", f"sign {sgn:+d} on {nm}",
                                 got, "-", res["ticks"], bool(ok)))

    def run_all(self):
        self.axis_moves()
        self.random_targets()
        self.below_table()
        self.single_rotations()
        self.combined_and_roundtrip()
        self.gripper()
        self.lift_descend_hold()
        self.atomic()
        self.go_home()
        self.done = self.ex.done()


def write_md(rep: Report, path: Path) -> dict:
    pe = [r.pos_err_mm for r in rep.rows if r.pos_err_mm is not None and r.passed]
    re_ = [r.rot_err_deg for r in rep.rows if r.rot_err_deg is not None and r.passed]
    stats = {"n": len(pe), "pos_mean_cm": np.mean(pe) / 10, "pos_worst_cm": np.max(pe) / 10,
             "rot_mean_deg": float(np.mean(re_)), "rot_worst_deg": float(np.max(re_))}
    n_pass = sum(r.passed for r in rep.rows)
    L = ["# Stage 3 primitive report (edge-slide, sim, no model)", "",
         f"**{n_pass}/{len(rep.rows)} checks passed** over seeds {sorted({r.seed for r in rep.rows})}.", "",
         "Commanded vs achieved error over passing MOVE_EEF-based moves "
         f"(n={stats['n']}): position mean **{stats['pos_mean_cm']:.2f} cm**, worst **{stats['pos_worst_cm']:.2f} cm**; "
         f"orientation mean **{stats['rot_mean_deg']:.2f} deg**, worst **{stats['rot_worst_deg']:.2f} deg**. "
         "Paper: 0.14 cm / 0.88 deg mean, 0.3 cm / 1.27 deg worst.", "",
         f"Gains: KP_POS={P.KP_POS}, KD_POS={P.KD_POS}s, KP_ROT={P.KP_ROT}, KD_ROT={P.KD_ROT}s, ROT_CAP={P.ROT_CAP}; "
         f"exit at {P.POS_TOL * 1000:.0f} mm / {P.ROT_TOL_DEG:.0f} deg and speed < {P.VEL_TOL} m/s, "
         f"{P.ANGVEL_TOL_DEG} deg/s for {P.SETTLE_TICKS} ticks. RPY = extrinsic XYZ (deg).", "",
         "Frames: 2x2 sheet, top = before, bottom = after; left = agentview (exo_right_cam), right = gripper_cam.", "",
         "| seed | group | test | target | achieved | error | steps | pass | notes | frames |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep.rows:
        fr = f"[img]({r.frames})" if r.frames else ""
        L.append(f"| {r.seed} | {r.group} | {r.name} | {r.target} | {r.achieved} | {r.error} | {r.steps} | "
                 f"{'PASS' if r.passed else '**FAIL**'} | {r.note} | {fr} |")
    path.write_text("\n".join(L) + "\n")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs/stage3_primitive_report")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = Report()
    for s in a.seeds:
        h = Harness(s, out, rep)
        try:
            h.run_all()
        finally:
            h.close()
    md = Path(str(out) + ".md")
    stats = write_md(rep, md)
    # frame links are relative to the report dir; write a copy inside it too
    (out / "report.md").write_text(md.read_text())
    md.write_text(md.read_text().replace("](frames/", f"]({out.name}/frames/"))
    print(f"\n{sum(r.passed for r in rep.rows)}/{len(rep.rows)} passed; stats {stats}\nreport: {md}")
    return rep, stats


if __name__ == "__main__":
    main()
