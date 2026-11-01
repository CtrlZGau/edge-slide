"""Stage 3 runner: closed-loop ReAct execution on edge-slide.

    python -m imbench.vlm_eval.run_stage3 --task edge-slide --seeds 0 --episodes 1 \
        --plan-source stage2 --stage2-sample 0 [--obj] [--mode primitives|code] [--budget-usd 20] [--mock]
        [--dry-run] [--render-prompts]

--mode primitives (default): one tool call = one of the 8 primitives.
--mode code: Code-as-Policies; each turn the model writes a Python program over the same primitives
             (sandboxed, see stage3/code_policy.py). --mock replays the reference program (needs --obj).

Output: runs/<model>/edge-slide/stage3[_code]/<plan-source>[-obj]/seed_<s>/ep_<e>/
  system_prompt.md tools.json steps.jsonl step<k>_prompt.json step<k>_raw.json frames/ episode_*.mp4 summary.json
Reasoning effort: Astra's LOWEST accepted effort ("low"; "minimal"/"none" are rejected by the API).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from imbench.vlm_eval.client import (
    LOWEST_EFFORT_CHAIN, MODEL, PRICE_PER_M_INPUT, PRICE_PER_M_OUTPUT, Ledger, VLMClient, cost_usd,
)
from imbench.vlm_eval.stage3.plan import load_plan

STAGE12_ROOT = "runs/gpt-6-astra/edge-slide"
SCRIPTED_RESULTS = "runs/stage3_scripted/results.json"

# dry-run token assumptions (measured: one 512x384 high-detail JPEG ~360 input tokens in the Stage-1 call)
TOK_IMG_HIGH = 360
TOK_IMG_LOW = 85            # OpenAI low-detail image cost; not measured for Astra
TOK_PER_MEMORY_ENTRY = 220
EST_OUT_PER_STEP = {"low": 400, "mid": 1200, "high": 3000}    # minimal-effort reasoning + tool args


def scripted_calls(seed: int) -> list[dict]:
    """The step-3 scripted primitive sequence for this seed, as mock tool calls."""
    res = [r for r in json.loads(Path(SCRIPTED_RESULTS).read_text()) if r["seed"] == seed]
    if not res:
        raise FileNotFoundError(f"no scripted sequence for seed {seed} in {SCRIPTED_RESULTS}")
    allowed = {"MOVE_EEF": ["position_m", "orientation_rpy_deg", "max_steps"], "LIFT": ["dz_m"],
               "DESCEND": ["target_z_m"], "HOLD": ["n_ticks"], "ATOMIC_ACTION": ["action7", "n_ticks"]}
    calls = []
    for s in res[0]["sequence"]:
        p = {k: v for k, v in s["params"].items() if k in allowed.get(s["primitive"], [])}
        if s["primitive"] == "MOVE_EEF":
            p.setdefault("orientation_rpy_deg", None)
            p.setdefault("max_steps", 400)
        calls.append({"name": s["primitive"], "args": {"params": p}})
    return calls


def ep_dir(model_dir: str, plan_source: str, obj: bool, seed: int, ep: int, mode: str = "primitives") -> Path:
    return Path("runs") / model_dir / "edge-slide" / ("stage3" if mode == "primitives" else "stage3_code") / \
        (plan_source + ("-obj" if obj else "")) / f"seed_{seed}" / f"ep_{ep}"


def render_step1(seed: int, plan_source: str, stage2_sample: int, obj: bool, out: Path, K: int, H: int,
                 mode: str = "primitives") -> dict:
    """Build (no API call) the step-1 prompt and save it as markdown. Returns token estimate inputs."""
    import imbench as ib
    from imbench.vlm_eval.stage3 import agent as A
    from imbench.vlm_eval.stage3 import primitives as P
    from imbench.vlm_eval.stage3.memory import FrameHistory, ReactMemory
    plan_text, meta = load_plan(plan_source, seed, stage2_sample, STAGE12_ROOT)
    task = ib.make("ib-p1-edge-slide-v1", obs_mode="combo", max_episode_steps=10_000)
    obs, _ = task.reset(seed=seed)
    ex = P.PrimitiveExecutor(task, obs, record_cam=None)
    views = A.save_views(ex, out / "frames", "step01")
    system = A.system_prompt(K, plan_text is not None, mode, obj)
    tail = "Choose the next primitive." if mode == "primitives" else "Write the next program (RUN_CODE), or call DONE."
    _, rec = A.build_user_content(1, K, plan_text, FrameHistory(H), A.state_summary(ex), A.camera_calibration(task),
                                  A.object_info(task) if obj else None, views, ReactMemory(H), 0, tail)
    task.close()
    md = ["# SYSTEM", "", system, "", "# USER (content parts in order)", ""]
    for c in rec:
        md.append(f"![{c.get('label')}]({c['file']})  _(image, detail={c['detail']})_" if c["type"] == "input_image"
                  else c["text"])
        md.append("")
    name = f"{'code_' if mode == 'code' else ''}{plan_source}{'-obj' if obj else ''}_seed{seed}_step1.md"
    (out / name).write_text("\n".join(md))
    text_chars = len(system) + sum(len(c.get("text", "")) for c in rec)
    return {"file": str(out / name), "text_chars": text_chars, "n_views": len(views), "meta": meta,
            "plan_text": plan_text}


def dry_run(seed, plan_source, stage2_sample, obj, K, H, episodes, mode="primitives"):
    out = Path("runs/gpt-6-astra/edge-slide/stage3/_prompt_preview")
    out.mkdir(parents=True, exist_ok=True)
    r = render_step1(seed, plan_source, stage2_sample, obj, out, K, H, mode)
    base_text = r["text_chars"] // 4
    if mode == "code":
        from imbench.vlm_eval.stage3.code_policy import code_tool_schemas
        tools_tok = len(json.dumps(code_tool_schemas())) // 4
        print("NOTE code mode: per-turn output (program text) and memory (past programs + logs) are larger than "
              "in primitives mode; the estimate below uses the primitives-mode per-step assumptions")
    else:
        tools_tok = len(json.dumps(__import__("imbench.vlm_eval.stage3.tools", fromlist=["x"]).tool_schemas())) // 4
    print(f"step-1 prompt: {r['file']}")
    print(f"text ~{base_text} tok (+ tool schemas ~{tools_tok}), {r['n_views']} current views x ~{TOK_IMG_HIGH} tok; "
          f"history adds {r['n_views']} views x ~{TOK_IMG_LOW} tok (low detail) per batch, memory ~{TOK_PER_MEMORY_ENTRY}/entry")
    rows = []
    for k in range(1, K + 1):
        h = min(k - 1, H)
        tin = base_text + tools_tok + r["n_views"] * TOK_IMG_HIGH + h * r["n_views"] * TOK_IMG_LOW + h * TOK_PER_MEMORY_ENTRY + h * 15
        rows.append(tin)
    for k in (1, 6, 13, 24):
        if k <= K:
            print(f"  step {k:2d}: input ~{rows[k - 1]} tok -> ${cost_usd(rows[k - 1], 0):.3f} input")
    for steps in (13, K):
        tin = sum(rows[:steps])
        cs = ", ".join(f"{lvl}-out ${cost_usd(tin, steps * o):.2f}" for lvl, o in EST_OUT_PER_STEP.items())
        print(f"  episode of {steps} steps: input ~{tin} tok; {cs}")
    tin = sum(rows)
    print(f"  x{episodes} episodes (worst, {K} steps each): ${episodes * cost_usd(tin, K * EST_OUT_PER_STEP['mid']):.2f} "
          f"at mid output; rates ${PRICE_PER_M_INPUT}/1M in, ${PRICE_PER_M_OUTPUT}/1M out")
    print(f"  output assumptions per step: {EST_OUT_PER_STEP} (minimal effort; unmeasured)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="edge-slide", choices=["edge-slide"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--plan-source", default="stage2", choices=["stage2", "oracle", "none"])
    ap.add_argument("--stage2-sample", type=int, default=0)
    ap.add_argument("--stage2-seed", type=int, default=None,
                    help="use the Stage-2 plan from this seed for every episode (default: the episode's own seed)")
    ap.add_argument("--obj", action="store_true", help="IMBAgent-obj: add object poses/extents (privileged)")
    ap.add_argument("--mode", default="primitives", choices=["primitives", "code"],
                    help="primitives: one primitive per call; code: Code-as-Policies (a program per call)")
    ap.add_argument("--budget-usd", type=float, default=20.0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--mock", action="store_true", help="replay the scripted step-3 sequence (no API)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--render-prompts", action="store_true", help="save step-1 prompt for every plan source")
    ap.add_argument("--in-process", action="store_true",
                    help="run episodes in this process (default: one subprocess per episode, because a second "
                         "MuJoCo env built in the same process renders identical, broken camera views)")
    ap.add_argument("--_child-episode", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.render_prompts:
        out = Path("runs/gpt-6-astra/edge-slide/stage3/_prompt_preview")
        for src in ("stage2", "oracle", "none"):
            for s in a.seeds:
                print(render_step1(s, src, a.stage2_sample, a.obj, out, a.K, a.H, a.mode)["file"])
        return 0
    if a.dry_run:
        dry_run(a.seeds[0], a.plan_source, a.stage2_sample, a.obj, a.K, a.H, a.episodes * len(a.seeds), a.mode)
        return 0

    from imbench.vlm_eval.stage3.agent import EpisodeConfig, MockToolClient, RenderError, run_episode
    model_dir = a.model + ("-MOCK" if a.mock else "")

    if a._child_episode is not None:                      # child: run exactly one episode, then exit
        seed, ep = a.seeds[0], a._child_episode
        plan_text, meta = load_plan(a.plan_source, seed, a.stage2_sample, STAGE12_ROOT, a.stage2_seed)
        ledger = Ledger(budget_usd=a.budget_usd)
        if a.mock and a.mode == "code":
            if not a.obj:
                raise SystemExit("--mock --mode code replays the reference program, which needs --obj")
            from imbench.vlm_eval.stage3.scripted_program import scripted_code_calls
            mock_calls = scripted_code_calls()
        else:
            mock_calls = scripted_calls(seed) if a.mock else None
        client = MockToolClient(mock_calls, ledger) if a.mock else \
            VLMClient(model=a.model, ledger=ledger, effort_chain=LOWEST_EFFORT_CHAIN)
        d = ep_dir(model_dir, a.plan_source, a.obj, seed, ep, a.mode)
        try:
            s = run_episode(client, EpisodeConfig(seed=seed, plan_text=plan_text, plan_meta=meta, obj=a.obj,
                                                  K=a.K, H=a.H, mode=a.mode), d)
        except RenderError as e:
            print(f"!! RENDER ERROR seed {seed} ep {ep}: {e}")
            return 3
        print(f"==> seed {seed} ep {ep}: success={s['success']} exit={s['exit_reason']} steps={s['steps']} "
              f"cost=${s['cost_usd']:.3f}")
        return 0

    import subprocess
    results, spent = [], 0.0
    for seed in a.seeds:
        load_plan(a.plan_source, seed, a.stage2_sample, STAGE12_ROOT, a.stage2_seed)   # fail fast if missing
        for ep in range(a.episodes):
            d = ep_dir(model_dir, a.plan_source, a.obj, seed, ep, a.mode)
            if (d / "summary.json").exists():
                results.append(json.loads((d / "summary.json").read_text()))
                print(f"seed {seed} ep {ep}: complete, skipping")
                continue
            remaining = a.budget_usd - spent
            if remaining <= 0:
                print(f"BUDGET STOP before seed {seed} ep {ep}: spent ${spent:.2f}")
                break
            print(f"seed {seed} ep {ep} [{a.mode} {a.plan_source}{' -obj' if a.obj else ''}"
                  f"{f' plan-seed {a.stage2_seed}' if a.stage2_seed is not None else ''}] -> {d}", flush=True)
            cmd = [sys.executable, "-m", "imbench.vlm_eval.run_stage3", "--seeds", str(seed), "--_child-episode",
                   str(ep), "--plan-source", a.plan_source, "--stage2-sample", str(a.stage2_sample),
                   "--budget-usd", f"{remaining:.4f}", "--model", a.model, "--K", str(a.K), "--H", str(a.H),
                   "--mode", a.mode]
            cmd += (["--stage2-seed", str(a.stage2_seed)] if a.stage2_seed is not None else [])
            cmd += (["--obj"] if a.obj else []) + (["--mock"] if a.mock else [])
            if a.in_process:
                main(cmd[3:])
            else:
                rc = subprocess.call(cmd)
                if rc not in (0,):
                    print(f"!! episode subprocess exit code {rc} (seed {seed} ep {ep})")
            if (d / "summary.json").exists():
                s = json.loads((d / "summary.json").read_text())
                spent += s["cost_usd"]
                results.append(s)
                if s["exit_reason"] == "budget":
                    break
    ledger = Ledger(budget_usd=a.budget_usd)
    ledger.spent_usd, ledger.calls = spent, sum(r["steps"] for r in results)
    n = len(results)
    sr = sum(r["success"] for r in results) / max(n, 1)
    print(f"\nSUCCESS {sum(r['success'] for r in results)}/{n} = {100 * sr:.0f}%   "
          f"TOTAL this run ${ledger.spent_usd:.2f} ({ledger.calls} calls, budget ${a.budget_usd:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
