"""Canned responses for --mock (offline pipeline test). Not model output."""

from __future__ import annotations

from imbench.vlm_eval.schemas import (
    AuditItem, Constraint, ConstraintArtifact, EpisodicObs, SelfAudit, Semantic,
    SimplestActionAnalysis, Stage2Plan, Subgoal, Unknown,
)


def stage1_artifact() -> ConstraintArtifact:
    return ConstraintArtifact(
        episodic=[EpisodicObs(camera="exo_right_cam", frame_idx=0, observation="MOCK: a flat box lies on a table"),
                  EpisodicObs(camera="ego_cam", frame_idx=0, observation="MOCK: the box is near the table centre")],
        semantic=Semantic(objects=["MOCK plate"], contacts=["plate on table"], reachability=["reachable"],
                          infeasibility=["MOCK"], hidden_state=["friction"], causal_preconditions=["MOCK"]),
        constraints=[
            Constraint(id="C1", statement="MOCK critical constraint", rank=1, criticality="critical",
                       evidence_refs=["cam:exo_right_cam:f0", "meta:controller_interface.end_effector"]),
            Constraint(id="C2", statement="MOCK minor constraint", rank=2, criticality="minor",
                       evidence_refs=["meta:sensor_snapshot.0.wrist_force_N"]),
        ],
        unknowns=[Unknown(what="MOCK friction", how_to_resolve_with_sensors="MOCK F/T while pushing")],
    )


def stage2_plan(score: float) -> Stage2Plan:
    a = lambda: AuditItem(score=score, reason=f"MOCK audit {score}")
    return Stage2Plan(
        simplest_action_analysis=SimplestActionAnalysis(
            simplest_action="MOCK direct grasp", why_it_fails="MOCK", violated_constraint_ids=["C1"]),
        subgoals=[Subgoal(id="S1", name="MOCK step", physical_outcome="MOCK", preconditions=[],
                          postconditions=["MOCK"], constraint_ids=["C1"])],
        self_audit=SelfAudit(F1_all_prerequisites_included=a(), F2_no_invalid_ordering=a(),
                             F3_no_dropped_information_gathering=a()),
    )


def canned(tag: str, schema, n: int):
    if schema is ConstraintArtifact:
        return stage1_artifact()
    if schema is Stage2Plan:
        # attempt 1 scores low to exercise the re-prompt loop; later attempts pass
        return stage2_plan(0.6 if tag.endswith("attempt1") else 0.9)
    raise ValueError(f"no canned response for {schema}")
