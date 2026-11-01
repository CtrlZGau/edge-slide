"""Stage 1: constraint understanding -> ConstraintArtifact (with quality gate and re-runs)."""

from __future__ import annotations

from typing import Callable

from imbench.vlm_eval.client import VLMClient, text_part
from imbench.vlm_eval.common import load_prompt, observation_content
from imbench.vlm_eval.schemas import ConstraintArtifact, gate_stage1

MAX_ATTEMPTS = 3


def build_prompt(bundle: dict, bundle_dir) -> tuple[str, list[dict], list[dict]]:
    system = load_prompt("stage1_system.md")
    parts, record = observation_content(bundle, bundle_dir)
    tail = "Follow the five steps in order and return the ConstraintArtifact."
    return system, parts + [text_part(tail)], record + [text_part(tail)]


def run(client: VLMClient, bundle: dict, bundle_dir, log: Callable[[dict], None]) -> dict:
    """Returns {artifact, gate_errors, passed, raws, prompt, attempts}."""
    system, parts, record = build_prompt(bundle, bundle_dir)
    raws, artifact, errs = [], None, ["not run"]
    for k in range(1, MAX_ATTEMPTS + 1):
        res = client.call(system, parts, ConstraintArtifact, tag=f"stage1_attempt{k}")
        raws.append(res.raw)
        if res.parsed is None:
            errs = [res.error or "schema parse failed"]
        else:
            artifact, errs = res.parsed, gate_stage1(res.parsed, bundle)
        log({"stage": 1, "attempt": k, "gate_passed": not errs, "gate_errors": errs,
             "usage": res.usage, "cost_usd": res.cost_usd, "model_snapshot": res.model_snapshot,
             "effort": res.effort})
        if not errs:
            break
    return {"artifact": artifact, "gate_errors": errs, "passed": not errs, "raws": raws,
            "prompt": {"system": system, "content": record}, "attempts": k}
