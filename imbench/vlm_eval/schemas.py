"""Pydantic schemas (Structured Outputs) and quality gates for Stage 1 / Stage 2."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── Stage 1 ──────────────────────────────────────────────────────────────

class EpisodicObs(_Strict):
    camera: str
    frame_idx: int
    observation: str


class Semantic(_Strict):
    objects: list[str]
    contacts: list[str]
    reachability: list[str]
    infeasibility: list[str]
    hidden_state: list[str]
    causal_preconditions: list[str]


class Constraint(_Strict):
    id: str
    statement: str
    rank: int
    criticality: Literal["critical", "important", "minor"]
    evidence_refs: list[str]


class Unknown(_Strict):
    what: str
    how_to_resolve_with_sensors: str


class ConstraintArtifact(_Strict):
    episodic: list[EpisodicObs]
    semantic: Semantic
    constraints: list[Constraint]
    unknowns: list[Unknown]


# ── Stage 2 ──────────────────────────────────────────────────────────────

class SimplestActionAnalysis(_Strict):
    simplest_action: str
    why_it_fails: str
    violated_constraint_ids: list[str]


class Subgoal(_Strict):
    id: str
    name: str
    physical_outcome: str
    preconditions: list[str]
    postconditions: list[str]
    constraint_ids: list[str]


class AuditItem(_Strict):
    score: float
    reason: str


class SelfAudit(_Strict):
    F1_all_prerequisites_included: AuditItem
    F2_no_invalid_ordering: AuditItem
    F3_no_dropped_information_gathering: AuditItem


class Stage2Plan(_Strict):
    simplest_action_analysis: SimplestActionAnalysis
    subgoals: list[Subgoal]
    self_audit: SelfAudit


# ── evidence references ──────────────────────────────────────────────────
# "cam:<camera>:f<idx>"   e.g. cam:exo_right_cam:f0
# "meta:<dotted.path>"    e.g. meta:controller_interface.end_effector, meta:sensor_snapshot.0.wrist_force_N

_CAM_RE = re.compile(r"^cam:([A-Za-z0-9_]+):f(\d+)$")


def _resolve_meta(metadata: dict, path: str) -> bool:
    node = metadata
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return False
    return True


def valid_frames(bundle: dict) -> set[tuple[str, int]]:
    return {(f["camera"], f["frame_idx"]) for f in bundle["frames"]}


def check_ref(ref: str, bundle: dict) -> bool:
    ref = ref.strip()
    m = _CAM_RE.match(ref)
    if m:
        return (m.group(1), int(m.group(2))) in valid_frames(bundle)
    if ref.startswith("meta:"):
        return _resolve_meta(bundle["metadata"], ref[len("meta:"):])
    return False


def gate_stage1(art: ConstraintArtifact, bundle: dict) -> list[str]:
    """Return a list of gate violations (empty = pass)."""
    errs = []
    if not art.constraints:
        errs.append("no constraints")
    frames = valid_frames(bundle)
    for e in art.episodic:
        if (e.camera, e.frame_idx) not in frames:
            errs.append(f"episodic entry cites unknown frame {e.camera}:f{e.frame_idx}")
    ids = [c.id for c in art.constraints]
    if len(set(ids)) != len(ids):
        errs.append(f"duplicate constraint ids: {ids}")
    for c in art.constraints:
        if not c.evidence_refs:
            errs.append(f"{c.id} has no evidence_refs")
        for r in c.evidence_refs:
            if not check_ref(r, bundle):
                errs.append(f"{c.id}: invalid evidence_ref {r!r}")
    return errs


def gate_stage2(plan: Stage2Plan, constraint_ids: set[str]) -> list[str]:
    errs = []
    if not plan.subgoals:
        errs.append("no subgoals")
    if not plan.simplest_action_analysis.why_it_fails.strip():
        errs.append("simplest_action_analysis.why_it_fails is empty")
    for cid in plan.simplest_action_analysis.violated_constraint_ids:
        if cid not in constraint_ids:
            errs.append(f"simplest_action_analysis cites unknown constraint {cid!r}")
    for sg in plan.subgoals:
        for cid in sg.constraint_ids:
            if cid not in constraint_ids:
                errs.append(f"subgoal {sg.id} cites unknown constraint {cid!r}")
    return errs


def audit_scores(plan: Stage2Plan) -> dict[str, float]:
    a = plan.self_audit
    clip = lambda x: max(0.0, min(1.0, float(x)))
    return {"F1": clip(a.F1_all_prerequisites_included.score),
            "F2": clip(a.F2_no_invalid_ordering.score),
            "F3": clip(a.F3_no_dropped_information_gathering.score)}


def audit_mean(plan: Stage2Plan) -> float:
    s = audit_scores(plan)
    return sum(s.values()) / len(s)
