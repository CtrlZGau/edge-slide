"""PLAN block for Stage 3: --plan-source stage2 | oracle | none."""

from __future__ import annotations

import json
from pathlib import Path

from imbench.vlm_eval.stage2 import gt_artifact

ORACLE_PLAN_TEXT = ("approach the plate, push it sideways until it overhangs the table edge, "
                    "pinch the exposed lip, and lift")

ORACLE_SUBGOALS = [
    {"id": "G1", "name": "Approach the plate",
     "physical_outcome": "The gripper is next to the plate, positioned to push it sideways toward a table edge.",
     "preconditions": ["The plate lies flat on the table, fully supported."],
     "postconditions": ["The gripper is in contact with, or just beside, the plate on the side away from the chosen edge."],
     "constraint_ids": ["C1", "C2"]},
    {"id": "G2", "name": "Push the plate sideways until it overhangs the table edge",
     "physical_outcome": "Part of the plate extends beyond the table edge, exposing a graspable lip, while the plate "
                         "is still supported by the table.",
     "preconditions": ["G1 achieved."],
     "postconditions": ["A lip of the plate overhangs the table edge.", "The plate has not fallen off the table."],
     "constraint_ids": ["C2"]},
    {"id": "G3", "name": "Pinch the exposed lip",
     "physical_outcome": "The jaws are closed on the overhanging lip, one finger above it and one below it.",
     "preconditions": ["G2 achieved: the lip overhangs enough for the finger pads."],
     "postconditions": ["The plate is held between the fingers."],
     "constraint_ids": ["C1", "C3"]},
    {"id": "G4", "name": "Lift",
     "physical_outcome": "The plate is held above the table.",
     "preconditions": ["G3 achieved."],
     "postconditions": ["The whole plate is clear of the table surface."],
     "constraint_ids": ["C3"]},
]


def _render(subgoals: list[dict], constraints: dict[str, str], header: str) -> str:
    L = [header, ""]
    for i, g in enumerate(subgoals, 1):
        L.append(f"{i}. [{g['id']}] {g['name']}")
        L.append(f"   physical outcome: {g['physical_outcome']}")
        L.append(f"   preconditions: {'; '.join(g['preconditions']) or '-'}")
        L.append(f"   postconditions: {'; '.join(g['postconditions']) or '-'}")
        if g.get("constraint_ids"):
            L.append(f"   constraints: {', '.join(g['constraint_ids'])}")
    used = sorted({c for g in subgoals for c in g.get("constraint_ids", [])},
                  key=lambda c: (len(c), c))
    if used:
        L += ["", "Constraints referenced by the plan:"]
        L += [f"- {c}: {constraints[c]}" for c in used if c in constraints]
    return "\n".join(L)


def load_plan(source: str, seed: int, stage2_sample: int = 0,
              stage12_root: str | Path = "runs/gpt-6-astra/edge-slide",
              plan_seed: int | None = None) -> tuple[str | None, dict]:
    """Return (plan_text or None, metadata). Raises if the stage2 plan is missing.

    plan_seed: take the Stage-2 plan from this seed instead of the episode seed (explicit override;
    the default requires the plan to come from the same scene)."""
    if source == "none":
        return None, {"plan_source": "none"}
    if source == "oracle":
        cons = {c.id: c.statement for c in gt_artifact().constraints}
        text = _render(ORACLE_SUBGOALS, cons, f"Canonical plan (oracle): {ORACLE_PLAN_TEXT}.")
        return text, {"plan_source": "oracle"}
    if source != "stage2":
        raise ValueError(f"unknown plan source {source!r}")

    root = Path(stage12_root)
    ps = seed if plan_seed is None else plan_seed
    d = root / f"seed_{ps}" / f"sample_{stage2_sample}"
    s2p, s1p, bp = d / "stage2.json", d / "stage1.json", root / f"seed_{ps}" / "bundle.json"
    for p in (s2p, s1p, bp):
        if not p.exists():
            raise FileNotFoundError(f"--plan-source stage2 needs {p} (Stage-2 plan for seed {ps}). "
                                    f"Run run_edge_slide --seeds {ps} first, or pass --stage2-seed.")
    bundle_seed = json.loads(bp.read_text())["seed"]
    if bundle_seed != ps:
        raise ValueError(f"Stage-2 bundle seed {bundle_seed} != requested plan seed {ps}")
    plan = json.loads(s2p.read_text())["plan"]
    if plan is None:
        raise ValueError(f"{s2p} has no plan")
    cons = {c["id"]: c["statement"] for c in json.loads(s1p.read_text())["artifact"]["constraints"]}
    header = (f"Plan produced by the Stage-2 planner for this scene (seed {seed}, sample {stage2_sample})."
              if ps == seed else
              f"Plan produced by the Stage-2 planner for another scene of this same task (seed {ps}, sample "
              f"{stage2_sample}). The plate's position, size and orientation in your scene may differ.")
    text = _render(plan["subgoals"], cons, header)
    return text, {"plan_source": "stage2", "stage2_path": str(s2p), "plan_seed": bundle_seed, "episode_seed": seed,
                  "stage2_sample": stage2_sample, "subgoal_ids": [g["id"] for g in plan["subgoals"]]}
