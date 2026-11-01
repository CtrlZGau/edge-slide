"""Stage 1 only, with gripper-aperture hints removed from the metadata ("blind" variant).

The normal bundle states the Panda's 8 cm max jaw opening and reports the finger
qpos / opening / openness fraction, which lets the model compare the jaw size with the
plate. This variant reuses the same frames but drops every numeric aperture fact, so
the model has to judge the plate geometry (and its thickness) from the images alone.

    python -m imbench.vlm_eval.run_stage1_blind --seeds 0 --samples 1 [--mock] [--budget-usd 10]

Output: runs/<model>/edge-slide-s1-blind/seed_X/{bundle.json,frames/}
        runs/<model>/edge-slide-s1-blind/seed_X/sample_k/{stage1_prompt.json,stage1_raw.json,
                                                       stage1.json,stage1_reasoning.md,attempts.jsonl,usage.json}
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from imbench.vlm_eval import stage1
from imbench.vlm_eval.client import MODEL, BudgetExceeded, Ledger, VLMClient
from imbench.vlm_eval.observation import load_or_build
from imbench.vlm_eval.reasoning import write_sample as write_reasoning

APERTURE_KEYS = ("gripper_finger_qpos_m", "gripper_opening_m", "gripper_openness_frac")


def blind_bundle(bundle: dict) -> dict:
    b = copy.deepcopy(bundle)
    md = b["metadata"]
    ci = md["controller_interface"]
    ci["end_effector"] = "parallel-jaw gripper (Panda hand), two fingers; no suction"
    for s in md["sensor_snapshot"]:
        for k in APERTURE_KEYS:
            s.pop(k, None)
        s["gripper_state"] = "fully open"
    b["variant"] = "s1-blind: no gripper aperture numbers (max opening, finger qpos, opening, openness)"
    return b


def assert_no_hints(bundle: dict) -> None:
    text = json.dumps(bundle["metadata"]).lower()
    for bad in ("0.08", "0.078", "0.039", "0.04", "8 cm", "max opening", "thick", "aperture", "openness_frac"):
        assert bad not in text, f"hint {bad!r} still in metadata"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--budget-usd", type=float, default=10.0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--runs-dir", default="runs")
    a = ap.parse_args(argv)

    ledger = Ledger(budget_usd=a.budget_usd)
    client = VLMClient(model=a.model, mock=a.mock, ledger=ledger)
    root = Path(a.runs_dir) / (a.model + ("-MOCK" if a.mock else "")) / "edge-slide-s1-blind"

    for seed in a.seeds:
        src = Path(a.runs_dir) / a.model / "edge-slide" / f"seed_{seed}"     # shared original bundle
        bundle = blind_bundle(load_or_build(seed, src))
        assert_no_hints(bundle)
        bdir = root / f"seed_{seed}"
        bdir.mkdir(parents=True, exist_ok=True)
        if not (bdir / "frames").exists():
            shutil.copytree(src / "frames", bdir / "frames")
        (bdir / "bundle.json").write_text(json.dumps(bundle, indent=2))

        for k in range(a.samples):
            d = bdir / f"sample_{k}"
            if (d / "usage.json").exists():
                print(f"seed {seed} sample {k}: complete, skipping")
                continue
            d.mkdir(parents=True, exist_ok=True)
            att = d / "attempts.jsonl"

            def log(rec):
                rec["time"] = datetime.now(timezone.utc).isoformat()
                with att.open("a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")

            try:
                r = stage1.run(client, bundle, bdir, log)
            except BudgetExceeded as e:
                print(f"BUDGET STOP at seed {seed} sample {k}: {e}")
                return 1
            (d / "stage1_prompt.json").write_text(json.dumps(r["prompt"], indent=2, default=str))
            (d / "stage1_raw.json").write_text(json.dumps(r["raws"], indent=2, default=str))
            (d / "stage1.json").write_text(json.dumps({
                "artifact": r["artifact"].model_dump() if r["artifact"] else None,
                "gate_passed": r["passed"], "gate_errors": r["gate_errors"], "attempts": r["attempts"]},
                indent=2, default=str))
            recs = [json.loads(l) for l in att.read_text().splitlines() if l.strip()]
            usage = {"calls": len(recs), "cost_usd": round(sum(x["cost_usd"] for x in recs), 4),
                     "efforts": sorted({str(x["effort"]) for x in recs})}
            (d / "usage.json").write_text(json.dumps(usage, indent=2))
            write_reasoning(d)
            print(f"seed {seed} sample {k}: gate_passed={r['passed']} attempts={r['attempts']} "
                  f"cost=${usage['cost_usd']:.2f}  -> {d}")
    print(f"TOTAL: {ledger.calls} calls, ${ledger.spent_usd:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
