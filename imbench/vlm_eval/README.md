# VLM eval: edge-slide (T01), GPT-6 Astra

Stage 1 (constraint understanding) + Stage 2 (high-level plan) from the paper, for one task.

## Commands (repo root, `.venv/bin/python`)

```bash
# key: OPENAI_API_KEY, or falls back to ~/openai_astra_key.txt
python -m imbench.vlm_eval.observation --seed 0 --out runs/gpt-6-astra/edge-slide   # bundle + contact sheet only
python -m imbench.vlm_eval.run_edge_slide --seeds 0 --samples 5 --dry-run           # token/cost estimate
python -m imbench.vlm_eval.run_edge_slide --seeds 0 --samples 5 --mock              # offline pipeline test
python -m imbench.vlm_eval.run_edge_slide --seeds 0 --samples 5 --budget-usd 30     # real run (resumable)
python -m imbench.vlm_eval.run_edge_slide --seeds 0 --samples 5 --gt                # S2-GT
python -m imbench.vlm_eval.grade_edge_slide                                         # hand grading -> grades.csv
python -m imbench.vlm_eval.grade_edge_slide --gt                                    # -> grades_s2gt.csv
python -m imbench.vlm_eval.grade_edge_slide --llm-pregrade --pregrade-model gpt-6-sol   # ASSIST ONLY
python -m pytest tests/test_vlm_eval.py -o addopts=""
```

Outputs: `runs/<model>/edge-slide/seed_X/` (bundle: `bundle.json`, `frames/`, `contact_sheet.png`),
`seed_X/sample_k/` (or `seed_X/s2gt/sample_k/`) with `frames/ stage1_prompt.json stage1_raw.json stage1.json
stage2_prompt.json stage2_raw.json stage2.json attempts.jsonl usage.json`. A sample is complete when
`usage.json` has `"complete": true`; complete samples are skipped on re-run. Mock runs go to
`runs/<model>-MOCK/`.

## Files
- `observation.py`: env reset, hold still, 1 frame x 4 cameras (512 px JPEG), non-privileged metadata.
- `client.py`: Responses API + Structured Outputs, retries, effort fallback, cost ledger, mock.
- `schemas.py`: pydantic schemas, evidence-ref checker, quality gates, audit score.
- `stage1.py`, `stage2.py`, `prompts/`: the two stages. Ground truth lives only in `stage2.gt_artifact()`.
- `rubrics/edge_slide.yaml`, `grade_edge_slide.py`: DRAFT rubric and interactive grader.

## Deviations from the paper
- **No temperature sweep.** GPT-6 Astra accepts no temperature/top_p, so the paper's 5 temperature
  settings are replaced by 5 independent samples at identical settings (recorded in `run_meta.json`).
- **Reasoning effort:** requests `max` (highest in openai SDK 3.42), falling back to `xhigh`/`high`
  only if the API rejects it; the effort actually used is logged per call. A reasoning summary is
  requested (dropped automatically if unsupported).
- **Rubric is a DRAFT** (no supplementary rubric in `docs/supplementary/`). S1.R1 is broadened: the
  sim plate is 3 cm thick and 8-15 cm wide against an 8 cm max jaw opening, so "too wide for the jaws
  while flush" is accepted as well as the paper's "too thin".
- **Prompts are our own** (no supplementary templates). The Stage-2 prompt uses the requested phrasing
  "e.g. a direct top-down grasp" for the simplest-action analysis.
- **Metadata includes the gripper's 8 cm max opening** (an embodiment fact, not an object pose).
- Stage 1 "re-run (max 3)" = at most 3 attempts in total. If all fail the gate, the last parseable
  artifact is kept (flagged `gate_passed: false`) and Stage 2 still runs.
- Stage 2: 6 attempts at most; the best is the highest audit mean among gate-passing plans.
- **One frame per camera** (t = 0.5 s) instead of 3: three frames at t = 0, 0.5, 1.0 s were near-identical
  (static scene, arm holding still), so they only added image tokens.
- Cost uses flat rates ($10/1M in, $50/1M out) and ignores cached-input discounts, so it over-estimates.
- The controller is OSC_POSE: the YAML says `IK`, which is not a valid type, so the base task falls back to OSC.

## Stage 3: closed-loop ReAct execution (paper App. F.5)

```bash
python -m imbench.vlm_eval.stage3.primitive_report --seeds 0 1 2          # primitive sim tests -> runs/stage3_primitive_report.md
python scripts/edge_slide_scripted_primitives.py --seeds 0 1 2 3 4 5 6 7 8 9 --video-seeds 0   # scripted feasibility
python -m imbench.vlm_eval.run_stage3 --seeds 0 --render-prompts [--obj]  # step-1 prompt for every plan source
python -m imbench.vlm_eval.run_stage3 --seeds 0 --episodes 10 --plan-source stage2 --dry-run
python -m imbench.vlm_eval.run_stage3 --seeds 0 --episodes 1 --plan-source stage2 --mock   # replay scripted seq
python -m imbench.vlm_eval.run_stage3 --task edge-slide --seeds 0 --episodes 1 \
    --plan-source stage2 --stage2-sample 0 [--obj] [--budget-usd 20]                       # real (Astra)
python -m pytest tests/test_primitives.py tests/test_stage3_agent.py -o addopts=""
```
Output: `runs/<model>/edge-slide/stage3/<plan-source>[-obj]/seed_<s>/ep_<e>/` with `system_prompt.md`,
`tools.json`, `steps.jsonl` (per step: reasoning, tool call, validity, executed primitive, return values,
state after, reward, cost, active sub-goal and status), `step<k>_prompt.json` (images by path),
`step<k>_raw.json`, `frames/`, `episode_<success|failure>.mp4`, `summary.json`. Finished episodes are skipped.

Files: `stage3/primitives.py` (8 primitives, GripperRegister, RPY helpers), `stage3/tools.py` (strict tool
schemas + validation), `stage3/agent.py` (observation, prompt slots, loop, mock client), `stage3/memory.py`,
`stage3/plan.py` (stage2 / oracle / none), `prompts/stage3_system.md` (robot reference block),
`stage3/primitive_report.py`, `run_stage3.py`, `scripts/edge_slide_scripted_primitives.py`.

### Stage-3 protocol deviations
- **Reasoning effort:** Stage 3 uses Astra's LOWEST accepted effort, `low` (the API accepts low/medium/high/xhigh/max;
  `minimal` and `none` are rejected), not `max`. The effort used is logged per step.
- **Pushing line of the robot reference block reworded.** In this env the table geom has friction 1.0, and
  MuJoCo uses max(geom friction), so plate-table friction is always 1.0, the same as the fingertip collision
  geoms. Pressing down on the top face and dragging never slides the plate (measured: 0-3 mm per pass).
  The prompt therefore says to push the plate's SIDE face with lowered fingertips, instead of the paper's
  "target slightly below the contact surface to keep downward pressure". The scripted solver uses the same
  side-face push.
- **Speeds in the prompt are measured, not `output_max`.** The OSC re-targets each tick to achieved + a*0.05 m,
  so the actual motion is ~0.003 m and ~0.6 deg per control step (10 cm in ~70 steps, 90 deg yaw in ~240).
  Both are stated.
- **MOVE_EEF** adds velocity damping (KD_POS 0.08 s, KD_ROT 0.1 s), caps the rotation command at 0.3, and
  only declares convergence once the arm is nearly still. **HOLD** and gripper actuation servo to a fixed pose,
  because a zero delta lets the arm drift.
- Commanded vs achieved error: 0.27 cm / 0.45 deg mean, 0.48 cm / 1.05 deg worst (paper: 0.14 cm / 0.88 deg,
  0.3 cm / 1.27 deg). Position error is higher because MOVE_EEF exits at the 5 mm tolerance.
- **Frame history:** the 12 history batches are sent downscaled (256 px, `detail: low`); current views are
  512 px, `detail: high`.
- LIFT/DESCEND have no max_steps parameter (default 400 steps). ATOMIC_ACTION and HOLD are capped at
  200 ticks. An invalid, off-schema or failed call executes HOLD(20) and is logged.
- **Contact flag:** |world-frame wrist force - its value at the start of the primitive| > 10 N, or any
  tactile reading > 0.
- **Episodes repeat the same seed:** the scene is identical across episodes of a seed (deterministic reset),
  and variation comes only from the model. This is needed because the Stage-2 plan is per seed.
- Side grasps are only reachable for the y = +-0.25 table edges; the far edge (x = +0.25) would need the
  hand to point back at the robot. This is not stated in the prompt.
- **Task text states the success height.** The env succeeds only when the plate centre z >= table + `lift_height`
  (0.80 + 0.15 = 0.95 m, read from the task YAML), with gripper contact. The paper's "Lift the plate above the
  table" led the model to stop at ~5 cm, so the task text is now "Lift the plate so that it is at least 15 cm
  above the table surface (plate centre z >= 0.95 m) while you hold it." Episodes run before this change are in
  `runs/gpt-6-astra/edge-slide/stage3_archive_v1_no_lift_height/`.
- **`--stage2-seed S`** uses the Stage-2 plan from seed S for every episode (the default still requires the
  episode's own seed). With an override, the PLAN header says the plan comes from another scene of the same task
  and that the plate may differ.
- **One subprocess per episode.** A second MuJoCo env built in the same process can render all four cameras as the
  same broken image (seen from step 3 of the second episode). `run_stage3` therefore runs each episode in a fresh
  subprocess (`--in-process` disables this), and the agent aborts an episode before its API call if all views are
  identical (`RenderError`).
- **Episode success comes from the env check only, not from the strategy.** Plates whose short side is close to the
  8 cm jaw opening (e.g. seed 2: 14.4 x 8.2 cm) can be wedged between the open jaws top-down and lifted without any
  edge slide. Such an episode counts as an env success but not as the intended strategy. Check `steps.jsonl` / the MP4
  before reporting.

## Stage 3, Code-as-Policies mode (`--mode code`)
Same task, observations (4 views + robot state, optional `--obj`), plan block, frame history, effort `low` and
turn budget `K` as the primitive mode. The only change: each turn the model calls `RUN_CODE` with a Python program
(or `DONE`) instead of a single primitive.
- **API in the program:** the same 8 primitives as functions (`move_eef`, `close_gripper`, `open_gripper`, `lift`,
  `descend`, `atomic_action`, `hold`, `done`), validated with the same pydantic validators. Read-outs: `get_state`,
  `eef_pos`, `eef_rpy`, `gripper_width`, `force_world`, `tactile`, `elapsed_steps`, plus `get_objects` only with
  `--obj`. Also `np`, `math` and `print`. The program cannot see images.
- **Returned to the model:** status (`ok` / `error` / `rejected` / `turn_limit` / `episode_stopped` / `done`), the
  error with its line, stdout, the per-call primitive log with return values, and the `result` variable. Past
  programs appear in working memory; older turns keep only the last 12 log entries.
- **Sandbox** (best effort, not a security boundary; each episode is its own subprocess): an AST check (no
  import, no `_` names or attributes, no exec/eval/open/getattr/type, no numpy file I/O attributes, no bare
  `except`), restricted builtins. Per-turn limits: 3000 control steps, 120 primitive calls, 600 s wall clock.
  The limits raise a `BaseException` subclass, so `except Exception` in model code can't swallow them. The
  program stops as soon as the env reports success.
- **Persistence:** one namespace per episode, so helpers defined in an early turn can be used later.
- **Output:** `runs/<model>/edge-slide/stage3_code/<plan>[-obj]/seed_<s>/ep_<e>/`.
- **Mock check:** `--mock --mode code --obj` replays a reference program (the scripted solver ported to the
  API, `stage3/scripted_program.py`, 3 turns). It got 3/3 on seeds 0–2. Tests are in `tests/test_code_policy.py`.
