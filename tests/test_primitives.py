"""Sim tests of the Stage-3 motor primitives on edge-slide (MuJoCo, no model).

Runs the same checks as `python -m imbench.vlm_eval.stage3.primitive_report` on one seed and
asserts every one passes. ~1-2 min.
"""

from __future__ import annotations

import numpy as np
import pytest

from imbench.vlm_eval.stage3 import primitives as P
from imbench.vlm_eval.stage3.primitive_report import Harness, Report

GROUPS = ["position", "orientation", "gripper", "lift/desc", "atomic"]


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    rep = Report()
    h = Harness(0, tmp_path_factory.mktemp("prim"), rep)
    try:
        h.run_all()
    finally:
        h.close()
    return rep


@pytest.mark.parametrize("group", GROUPS)
def test_primitive_group(report, group):
    rows = [r for r in report.rows if r.group == group]
    assert rows, f"no rows for {group}"
    failed = [f"{r.name}: err {r.error} {r.note}" for r in rows if not r.passed]
    assert not failed, "\n".join(failed)


def test_commanded_vs_achieved(report):
    pe = [r.pos_err_mm for r in report.rows if r.pos_err_mm is not None]
    re_ = [r.rot_err_deg for r in report.rows if r.rot_err_deg is not None]
    assert max(pe) < 5.0 and max(re_) < 5.0
    print(f"pos mean {np.mean(pe):.2f} mm worst {max(pe):.2f}; rot mean {np.mean(re_):.2f} worst {max(re_):.2f} deg")


# ── pure-python checks (no sim) ──────────────────────────────────────────

@pytest.mark.parametrize("rpy", [[180, 0, 0], [-178.8, -8.2, 0.3], [170, 15, -90], [0, 0, 45], [30, -60, 120]])
def test_rpy_round_trip(rpy):
    q = P.rpy_deg_to_rot(rpy).as_quat()                   # xyzw
    back = P.quat_to_rpy_deg(q)
    assert np.degrees((P.rpy_deg_to_rot(back) * P.rpy_deg_to_rot(rpy).inv()).magnitude()) < 1e-6


def test_rpy_is_extrinsic_xyz():
    from scipy.spatial.transform import Rotation as R
    r = P.rpy_deg_to_rot([10, 20, 30])
    manual = R.from_euler("z", 30, degrees=True) * R.from_euler("y", 20, degrees=True) * R.from_euler("x", 10, degrees=True)
    assert (r * manual.inv()).magnitude() < 1e-9


def test_rot_error_world_frame():
    from scipy.spatial.transform import Rotation as R
    cur = P.rpy_deg_to_rot([180, 0, 0])
    tgt = R.from_euler("z", 30, degrees=True) * cur            # +30 deg about WORLD z
    e = P.rot_error_vec(tgt, cur)
    assert np.allclose(e, [0, 0, np.radians(30)], atol=1e-9)


def test_gripper_register():
    g = P.GripperRegister()
    assert g.value == -1.0
    g.set(3)
    assert g.value == 1.0
    g.set(-0.4)
    assert g.value == -0.4
