"""Stage 2: high-level plan conditioned on the (binding) Stage-1 artifact, with self-audit loop.

--gt mode (paper Table 2 "S2-GT") replaces the Stage-1 artifact with gt_artifact(),
built from the task's ground-truth Intuition / Canonical-plan text.
"""

from __future__ import annotations

import json
from typing import Callable

from imbench.vlm_eval.client import VLMClient, text_part
from imbench.vlm_eval.common import load_prompt, observation_content
from imbench.vlm_eval.schemas import (
    Constraint, ConstraintArtifact, Semantic, Stage2Plan, audit_mean, audit_scores, gate_stage2,
)

MAX_ATTEMPTS = 6
AUDIT_THRESHOLD = 0.8

# Ground truth (paper App. B.2). Used ONLY for --gt and grading; never in normal prompts.
GT_INTUITION = ("A top-down jaw closure cannot close around this small thickness; the plate must first "
                "be slid to overhang the table edge to expose a graspable lip.")
GT_CANONICAL_PLAN = ("Approach the plate, push it sideways until it overhangs the table edge, pinch the "
                     "exposed lip, and lift.")


def gt_artifact() -> ConstraintArtifact:
    ref = ["meta:task_description"]
    return ConstraintArtifact(
        episodic=[],
        semantic=Semantic(
            objects=["thin rectangular plate lying flush on the table", "table with an accessible edge"],
            contacts=["plate rests flat on the table top; no clearance under it"],
            reachability=["the plate and the nearby table edge are within the arm's workspace"],
            infeasibility=["a top-down jaw closure cannot close around the plate's small thickness"],
            hidden_state=[],
            causal_preconditions=["the plate must overhang the table edge before a lip can be pinched"],
        ),
        constraints=[
            Constraint(id="C1", rank=1, criticality="critical", evidence_refs=ref,
                       statement="A direct top-down jaw grasp is infeasible: the jaws cannot close around "
                                 "the thin plate lying flush on the table."),
            Constraint(id="C2", rank=2, criticality="critical", evidence_refs=ref,
                       statement="The plate must first be slid sideways until it overhangs the table edge, "
                                 "exposing a graspable lip."),
            Constraint(id="C3", rank=3, criticality="important", evidence_refs=ref,
                       statement="The grasp must pinch the exposed lip; only then can the plate be lifted."),
        ],
        unknowns=[],
    )


def build_prompt(artifact: ConstraintArtifact, bundle: dict, bundle_dir, feedback: str | None):
    system = load_prompt("stage2_system.md")
    parts, record = observation_content(bundle, bundle_dir)
    art = ("CONSTRAINT ARTIFACT FROM STAGE 1 (verbatim JSON; its constraints are binding):\n"
           + artifact.model_dump_json(indent=2))
    tail = ("Analyse the simplest action first, then give the ordered sub-goals, then the self-audit."
            + (f"\n\n{feedback}" if feedback else ""))
    extra = [text_part(art)]
    return system, extra + parts + [text_part(tail)], extra + record + [text_part(tail)]


def run(client: VLMClient, artifact: ConstraintArtifact, bundle: dict, bundle_dir,
        log: Callable[[dict], None]) -> dict:
    ids = {c.id for c in artifact.constraints}
    retry_tpl = load_prompt("stage2_retry.md")
    best, best_score, best_k, raws, prompts, feedback = None, -1.0, 0, [], [], None
    for k in range(1, MAX_ATTEMPTS + 1):
        system, parts, record = build_prompt(artifact, bundle, bundle_dir, feedback)
        prompts.append({"attempt": k, "system": system, "content": record})
        res = client.call(system, parts, Stage2Plan, tag=f"stage2_attempt{k}")
        raws.append(res.raw)
        plan, errs, score, scores = res.parsed, [], -1.0, None
        if plan is None:
            errs = [res.error or "schema parse failed"]
        else:
            errs = gate_stage2(plan, ids)
            scores, score = audit_scores(plan), audit_mean(plan)
            eff = score if not errs else -0.5 + score / 10   # gate-failing plans rank below passing ones
            if eff > best_score:
                best, best_score, best_k = plan, eff, k
        log({"stage": 2, "attempt": k, "gate_passed": plan is not None and not errs, "gate_errors": errs,
             "audit": scores, "audit_mean": score, "usage": res.usage, "cost_usd": res.cost_usd,
             "model_snapshot": res.model_snapshot, "effort": res.effort})
        if plan is not None and not errs and score >= AUDIT_THRESHOLD:
            break
        if plan is not None:
            feedback = retry_tpl.format(mean=score, threshold=AUDIT_THRESHOLD,
                                        previous=plan.model_dump_json(indent=2)
                                        + (f"\n\nGATE ERRORS: {json.dumps(errs)}" if errs else ""))
    return {"plan": best, "best_attempt": best_k, "audit_mean": audit_mean(best) if best else None,
            "accepted": best is not None and best_score >= AUDIT_THRESHOLD, "attempts": k,
            "raws": raws, "prompts": prompts}
