"""Interactive hand-grading of edge-slide samples against rubrics/edge_slide.yaml.

    python -m imbench.vlm_eval.grade_edge_slide [--run-dir runs/gpt-6-astra/edge-slide] [--gt]
                                                [--llm-pregrade --pregrade-model gpt-6-sol]

Stage-1 pass = every REQUIRED item yes and every FORBIDDEN item no.
Stage-2 pass (CORRECT) = every criterion yes and every FORBIDDEN item no.
--llm-pregrade pre-fills suggestions from a DIFFERENT model. ASSIST ONLY: you still answer
every item (Enter accepts the suggestion).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

RUBRIC = Path(__file__).parent / "rubrics" / "edge_slide.yaml"


def load_rubric(path=RUBRIC) -> dict:
    return yaml.safe_load(Path(path).read_text())


def stage_items(rubric: dict, gt: bool) -> list[tuple[str, str, str]]:
    """(stage, kind, id/text) list in asking order. kind in required/optional/forbidden/criteria."""
    out = []
    if not gt:
        for kind in ("required", "optional", "forbidden"):
            out += [("stage1", kind, it["id"], it["text"]) for it in rubric["stage1"][kind]]
    for kind in ("criteria", "forbidden"):
        out += [("stage2", kind, it["id"], it["text"]) for it in rubric["stage2"][kind]]
    return out


def stage1_pass(rubric: dict, ans: dict[str, bool]) -> bool:
    s = rubric["stage1"]
    return all(ans[i["id"]] for i in s["required"]) and not any(ans[i["id"]] for i in s["forbidden"])


def stage2_pass(rubric: dict, ans: dict[str, bool]) -> bool:
    s = rubric["stage2"]
    return all(ans[i["id"]] for i in s["criteria"]) and not any(ans[i["id"]] for i in s["forbidden"])


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = 100 * sum(xs) / len(xs)
    sd = 100 * statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, sd


# ── display ───────────────────────────────────────────────────────────────

def fmt_stage1(a: dict) -> str:
    L = ["STAGE 1 ARTIFACT"]
    for e in a["episodic"]:
        L.append(f"  [ep {e['camera']}:f{e['frame_idx']}] {e['observation']}")
    for k, v in a["semantic"].items():
        L.append(f"  {k}: " + ("; ".join(v) if v else "-"))
    for c in sorted(a["constraints"], key=lambda c: c["rank"]):
        L.append(f"  {c['id']} (rank {c['rank']}, {c['criticality']}): {c['statement']}  <{', '.join(c['evidence_refs'])}>")
    for u in a["unknowns"]:
        L.append(f"  unknown: {u['what']}  -> {u['how_to_resolve_with_sensors']}")
    return "\n".join(L)


def fmt_stage2(p: dict | None) -> str:
    if p is None:
        return "STAGE 2: no plan"
    s = p["simplest_action_analysis"]
    L = ["STAGE 2 PLAN",
         f"  simplest action: {s['simplest_action']}",
         f"  why it fails: {s['why_it_fails']}  [{', '.join(s['violated_constraint_ids'])}]"]
    for g in p["subgoals"]:
        L.append(f"  {g['id']} {g['name']}: {g['physical_outcome']}  [{', '.join(g['constraint_ids'])}]")
        L.append(f"      pre: {'; '.join(g['preconditions']) or '-'} | post: {'; '.join(g['postconditions']) or '-'}")
    for k, v in p["self_audit"].items():
        L.append(f"  audit {k}: {v['score']:.2f}  {v['reason']}")
    return "\n".join(L)


# ── LLM pre-grade (ASSIST ONLY) ──────────────────────────────────────────

class _Verdict(BaseModel):
    item_id: str
    verdict: Literal["y", "n"]
    reason: str


class _Pregrade(BaseModel):
    verdicts: list[_Verdict]


def llm_pregrade(model: str, items, s1: dict, s2: dict | None) -> dict[str, tuple[str, str]]:
    from imbench.vlm_eval.client import VLMClient, text_part
    client = VLMClient(model=model)
    rubric_txt = "\n".join(f"{i} ({stage} {kind}): {text}" for stage, kind, i, text in items)
    system = ("You are grading a robot-planning model's outputs against a rubric. For each rubric item answer "
              "'y' if the item is satisfied/present in the output and 'n' otherwise (for FORBIDDEN items, 'y' "
              "means the forbidden thing IS present). Judge only what is written. Give a one-sentence reason.")
    content = [text_part(f"RUBRIC:\n{rubric_txt}\n\nSTAGE 1 OUTPUT:\n{json.dumps(s1, indent=2)}\n\n"
                         f"STAGE 2 OUTPUT:\n{json.dumps(s2, indent=2)}")]
    res = client.call(system, content, _Pregrade, tag="pregrade")
    print(f"  [pregrade cost ${res.cost_usd:.3f}, model {res.model_snapshot}]")
    return {v.item_id: (v.verdict, v.reason) for v in res.parsed.verdicts} if res.parsed else {}


# ── main ─────────────────────────────────────────────────────────────────

def ask(prompt: str, default: str | None) -> bool:
    hint = f" [suggest {default}]" if default else ""
    while True:
        a = input(f"{prompt}{hint} (y/n): ").strip().lower() or (default or "")
        if a in ("y", "n"):
            return a == "y"


def samples(run_dir: Path, gt: bool):
    for seed_dir in sorted(run_dir.glob("seed_*")):
        base = seed_dir / "s2gt" if gt else seed_dir
        for d in sorted(base.glob("sample_*"), key=lambda p: int(p.name.split("_")[1])):
            u = d / "usage.json"
            if u.exists() and json.loads(u.read_text()).get("complete"):
                yield seed_dir, d


def summarize(rows: list[dict], rubric: dict, gt: bool) -> None:
    s1 = [float(r["stage1_pass"]) for r in rows if r.get("stage1_pass") not in ("", None)]
    s2 = [float(r["stage2_pass"]) for r in rows]
    print(f"\n=== edge-slide, {len(rows)} graded samples (rubric {rubric.get('status', '')}) ===")
    if not gt:
        m, sd = mean_std(s1)
        print(f"Stage 1: {m:.1f} ± {sd:.1f}")
    m, sd = mean_std(s2)
    print(f"Stage 2{' (S2-GT)' if gt else ''}: {m:.1f} ± {sd:.1f}")
    print("Paper reference:", ", ".join(
        f"{k} " + "/".join(f"{st} {v:.1f}" for st, v in ref.items()) for k, ref in rubric["reference"].items()))
    print("(std is the sample std over samples, in percentage points)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="runs/gpt-6-astra/edge-slide")
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--llm-pregrade", action="store_true", help="ASSIST ONLY suggestions from another model")
    ap.add_argument("--pregrade-model", default="gpt-6-sol")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)

    run_dir = Path(a.run_dir)
    rubric = load_rubric()
    items = stage_items(rubric, a.gt)
    out = run_dir / ("grades_s2gt.csv" if a.gt else "grades.csv")
    fields = ["seed", "sample"] + [i for _, _, i, _ in items] + ["stage1_pass", "stage2_pass", "pregrade_model"]
    rows = list(csv.DictReader(out.open())) if out.exists() else []
    done = {(r["seed"], r["sample"]) for r in rows}

    if not a.summary_only:
        if a.llm_pregrade:
            print(f"*** LLM PRE-GRADE ENABLED ({a.pregrade_model}): ASSIST ONLY, you make every call ***")
        for seed_dir, d in samples(run_dir, a.gt):
            key = (seed_dir.name, d.name)
            if key in done:
                continue
            s1 = json.loads((d / "stage1.json").read_text())
            s2 = json.loads((d / "stage2.json").read_text())
            print("\n" + "=" * 100)
            print(f"{seed_dir.name}/{d.name}   contact sheet: {seed_dir / 'contact_sheet.png'}")
            print(f"stage-1 source: {s1.get('source')}  gate_passed={s1.get('gate_passed')}  "
                  f"| stage-2 attempts={s2['attempts']} best={s2['best_attempt']} audit_mean={s2['audit_mean']}")
            print(fmt_stage1(s1["artifact"]))
            print(fmt_stage2(s2["plan"]))
            sugg = llm_pregrade(a.pregrade_model, items, s1["artifact"], s2["plan"]) if a.llm_pregrade else {}
            ans = {}
            for stage, kind, iid, text in items:
                v = sugg.get(iid)
                if v:
                    print(f"    ASSIST ONLY ({a.pregrade_model}): {v[0]} - {v[1]}")
                ans[iid] = ask(f"  {iid} [{stage} {kind.upper()}] {text}", v[0] if v else None)
            row = {"seed": key[0], "sample": key[1], **{k: int(v) for k, v in ans.items()},
                   "stage1_pass": "" if a.gt else int(stage1_pass(rubric, ans)),
                   "stage2_pass": int(stage2_pass(rubric, ans)),
                   "pregrade_model": a.pregrade_model if a.llm_pregrade else ""}
            print(f"  -> stage1_pass={row['stage1_pass']} stage2_pass={row['stage2_pass']}")
            rows.append(row)
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
    print(f"grades: {out}")
    summarize(rows, rubric, a.gt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
