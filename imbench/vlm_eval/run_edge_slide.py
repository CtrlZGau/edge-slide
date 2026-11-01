"""Run the Stage-1 + Stage-2 VLM eval on edge-slide.

    python -m imbench.vlm_eval.run_edge_slide --seeds 0 --samples 5 [--gt] [--mock] [--dry-run] [--budget-usd 30]

Layout: runs/<model>/edge-slide/seed_X/{bundle.json,frames/,contact_sheet.png}
        runs/<model>/edge-slide/seed_X/sample_k/          (normal)
        runs/<model>/edge-slide/seed_X/s2gt/sample_k/     (--gt)
GPT-6 Astra has no temperature, so the paper's 5 temperature settings are replaced by
5 independent samples at the same settings (recorded in run_meta.json).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from imbench.vlm_eval import stage1, stage2
from imbench.vlm_eval.client import (
    EFFORT_CHAIN, MODEL, PRICE_PER_M_INPUT, PRICE_PER_M_OUTPUT, BudgetExceeded, Ledger, VLMClient, cost_usd,
)
from imbench.vlm_eval.observation import load_or_build
from imbench.vlm_eval.reasoning import write_sample as write_reasoning
from imbench.vlm_eval.schemas import ConstraintArtifact

TASK_DIR = "edge-slide"

# dry-run assumptions (conservative)
EST_TOKENS_PER_IMAGE = 765          # upper end of OpenAI high-detail image cost for a 512x384 JPEG
EST_OUTPUT_PER_CALL = {"low": 4000, "mid": 12000, "high": 30000}  # reasoning at max effort dominates


def _w(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sample_dir(root: Path, seed: int, k: int, gt: bool) -> Path:
    return root / f"seed_{seed}" / ("s2gt" if gt else "") / f"sample_{k}"


def is_complete(d: Path) -> bool:
    u = d / "usage.json"
    return u.exists() and json.loads(u.read_text()).get("complete", False)


def run_sample(client: VLMClient, bundle: dict, bundle_dir: Path, d: Path, gt: bool) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "frames").exists():
        shutil.copytree(bundle_dir / "frames", d / "frames")
    attempts_f = d / "attempts.jsonl"

    def log(rec):
        rec["time"] = datetime.now(timezone.utc).isoformat()
        with attempts_f.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # Stage 1 (resumable: reuse stage1.json if present)
    if (d / "stage1.json").exists():
        artifact = ConstraintArtifact.model_validate(json.loads((d / "stage1.json").read_text())["artifact"])
    elif gt:
        artifact = stage2.gt_artifact()
        _w(d / "stage1_prompt.json", {"note": "S2-GT: no Stage-1 call; ground-truth artifact used"})
        _w(d / "stage1_raw.json", [])
        _w(d / "stage1.json", {"artifact": artifact.model_dump(), "source": "ground_truth",
                               "gate_passed": True, "gate_errors": [], "attempts": 0})
    else:
        r1 = stage1.run(client, bundle, bundle_dir, log)
        _w(d / "stage1_prompt.json", r1["prompt"])
        _w(d / "stage1_raw.json", r1["raws"])
        if r1["artifact"] is None:
            raise RuntimeError(f"Stage 1 produced no parseable artifact: {r1['gate_errors']}")
        artifact = r1["artifact"]
        _w(d / "stage1.json", {"artifact": artifact.model_dump(), "source": "model",
                               "gate_passed": r1["passed"], "gate_errors": r1["gate_errors"],
                               "attempts": r1["attempts"]})

    # Stage 2
    r2 = stage2.run(client, artifact, bundle, bundle_dir, log)
    _w(d / "stage2_prompt.json", r2["prompts"])
    _w(d / "stage2_raw.json", r2["raws"])
    _w(d / "stage2.json", {"plan": r2["plan"].model_dump() if r2["plan"] else None,
                           "best_attempt": r2["best_attempt"], "audit_mean": r2["audit_mean"],
                           "accepted": r2["accepted"], "attempts": r2["attempts"]})

    # usage (all calls ever made for this sample, including interrupted earlier runs)
    recs = [json.loads(l) for l in attempts_f.read_text().splitlines() if l.strip()]
    tin = sum(r["usage"].get("input_tokens", 0) for r in recs)
    tout = sum(r["usage"].get("output_tokens", 0) for r in recs)
    reasoning = sum((r["usage"].get("output_tokens_details") or {}).get("reasoning_tokens", 0) for r in recs)
    cached = sum((r["usage"].get("input_tokens_details") or {}).get("cached_tokens", 0) for r in recs)
    usage = {"complete": True, "calls": len(recs), "input_tokens": tin, "cached_input_tokens": cached,
             "output_tokens": tout, "reasoning_tokens": reasoning,
             "cost_usd": round(sum(r["cost_usd"] for r in recs), 4),
             "model_snapshots": sorted({r["model_snapshot"] for r in recs}),
             "efforts": sorted({str(r["effort"]) for r in recs})}
    write_reasoning(d)
    _w(d / "usage.json", usage)
    return usage


def estimate(bundle: dict, bundle_dir: Path, gt: bool, n_samples: int) -> None:
    from imbench.vlm_eval.common import load_prompt
    n_img = len(bundle["frames"])
    _, parts, _ = stage1.build_prompt(bundle, bundle_dir)
    text_chars = sum(len(p.get("text", "")) for p in parts) + len(load_prompt("stage1_system.md"))
    s1_in = n_img * EST_TOKENS_PER_IMAGE + text_chars // 4
    artifact_tokens = 2500
    s2_in = s1_in + artifact_tokens + 500
    s2_retry_in = s2_in + 2000
    print(f"images/call: {n_img} x ~{EST_TOKENS_PER_IMAGE} tok; stage-1 input ~{s1_in} tok; "
          f"stage-2 input ~{s2_in} tok (+~2000 per retry)")
    scen = {"typical (S1 x1, S2 x2)": (0 if gt else 1, 2), "worst (S1 x3, S2 x6)": (0 if gt else 3, 6)}
    for name, (n1, n2) in scen.items():
        tin = n1 * s1_in + s2_in + (n2 - 1) * s2_retry_in
        row = []
        for lvl, out in EST_OUTPUT_PER_CALL.items():
            row.append(f"{lvl}-output ${cost_usd(tin, (n1 + n2) * out):.2f}")
        print(f"  per sample, {name}: input ~{tin} tok; " + ", ".join(row))
    print(f"  output assumptions per call: {EST_OUTPUT_PER_CALL} tokens (reasoning tokens are billed as output)")
    print(f"  rates: ${PRICE_PER_M_INPUT}/1M input, ${PRICE_PER_M_OUTPUT}/1M output; "
          f"x{n_samples} samples per seed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--gt", action="store_true", help="S2-GT: Stage 2 on the ground-truth artifact")
    ap.add_argument("--mock", action="store_true", help="canned responses, no API calls")
    ap.add_argument("--dry-run", action="store_true", help="estimate tokens/cost only")
    ap.add_argument("--budget-usd", type=float, default=30.0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--runs-dir", default="runs")
    a = ap.parse_args(argv)

    model_dir = a.model + ("-MOCK" if a.mock else "")
    root = Path(a.runs_dir) / model_dir / TASK_DIR
    root.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(budget_usd=a.budget_usd)

    if a.dry_run:
        for seed in a.seeds:
            bdir = Path(a.runs_dir) / a.model / TASK_DIR / f"seed_{seed}"
            bundle = load_or_build(seed, bdir)
            todo = [k for k in range(a.samples) if not is_complete(sample_dir(root, seed, k, a.gt))]
            print(f"seed {seed}: {len(todo)}/{a.samples} samples to run ({'S2-GT' if a.gt else 'S1+S2'})")
            estimate(bundle, bdir, a.gt, len(todo))
        return 0

    client = VLMClient(model=a.model, mock=a.mock, ledger=ledger)
    _w(root / "run_meta.json", {
        "model": a.model, "mock": a.mock, "reasoning_effort_requested": EFFORT_CHAIN[0],
        "effort_fallback_chain": list(EFFORT_CHAIN), "temperature": "not supported by model; not sent",
        "sampling": "paper averages over 5 temperature settings; here N independent samples at identical settings",
        "samples": a.samples, "seeds": a.seeds, "gt_mode": a.gt, "git_head": _git_head(),
        "prices_usd_per_1M": {"input": PRICE_PER_M_INPUT, "output": PRICE_PER_M_OUTPUT},
        "started": datetime.now(timezone.utc).isoformat()})

    stopped = False
    for seed in a.seeds:
        bdir = Path(a.runs_dir) / a.model / TASK_DIR / f"seed_{seed}"   # shared bundle
        bundle = load_or_build(seed, bdir)
        for k in range(a.samples):
            d = sample_dir(root, seed, k, a.gt)
            if is_complete(d):
                print(f"seed {seed} sample {k}: complete, skipping")
                continue
            try:
                u = run_sample(client, bundle, bdir, d, a.gt)
                print(f"seed {seed} sample {k}: done  calls={u['calls']} cost=${u['cost_usd']:.2f}  "
                      f"[{json.loads((d / 'stage2.json').read_text())['attempts']} stage-2 attempts]")
            except BudgetExceeded as e:
                print(f"BUDGET STOP at seed {seed} sample {k}: {e}. Sample left incomplete; re-run to resume.")
                stopped = True
                break
        if stopped:
            break
    print(f"TOTAL this run: {ledger.calls} API calls, ${ledger.spent_usd:.2f} "
          f"(budget ${a.budget_usd:.2f}){' [MOCK]' if a.mock else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
