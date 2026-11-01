# VLM eval on edge-slide (GPT-6 Astra): transfer summary

Everything needed to move the VLM-eval work (Stage 1, Stage 2, Stage 3 primitives, Stage 3 Code-as-Policies)
out of IMBench into its own repo. Written 2026-11-01. `README.md` in this folder holds the command reference and
the full list of protocol deviations; this file covers what to copy, what it depends on, how it works, and every
result so far.

---

## 1. What this is

An evaluation of the VLM **GPT-6 Astra** (OpenAI Responses API) on IMBench task **T01 edge-slide**
(`ib-p1-edge-slide-v1`). The task: a flat plate (3 cm thick, 8–15 cm wide, 10–20 cm long) lies on a table and is
too wide for the 8 cm gripper. The intended solution is to slide it until it overhangs a table edge, pinch the
overhang from the side, and lift.

| Stage | What the model does | Effort | Interface |
|---|---|---|---|
| 1. Constraint understanding | identifies the physical constraints from 4 camera frames + metadata | `max` | Structured Outputs (pydantic) |
| 2. High-level plan | writes a sub-goal plan (pre/post-conditions, constraint ids); audited against ground truth | `max` | Structured Outputs |
| 3a. Closed-loop ReAct (primitives) | calls ONE of 8 motor primitives per turn, sees the result, repeats | `low` | function calling (strict tools) |
| 3b. Closed-loop Code-as-Policies | writes ONE Python program per turn over the same primitives + read-outs | `low` | function calling (`RUN_CODE`, `DONE`) |

The env code (IMBench task classes, YAML) was **not modified** by this work.

---

## 2. Files to copy

All files are untracked in IMBench git (nothing committed). Line counts are current.

### Code: `imbench/vlm_eval/` (the whole folder)
| File | Lines | Role |
|---|---|---|
| `__init__.py` | 1 | package marker |
| `client.py` | ~250 | OpenAI Responses client: `call()` (Structured Outputs) and `call_tools()` (function calling), effort fallback chains, retries, `Ledger` budget, `BudgetExceeded`, `QuotaExhausted`, cost at $10/1M in and $50/1M out, mock mode, API key lookup |
| `common.py` | 35 | `load_prompt`, `observation_content` (Stage 1/2 multimodal input) |
| `observation.py` | 157 | `build_bundle(seed)`: env reset, hold still, 1 frame × 4 cameras at t = 0.5 s (512 px JPEG), non-privileged metadata, contact sheet |
| `schemas.py` | 166 | pydantic `ConstraintArtifact`, `Stage2Plan`; evidence-ref checker (`cam:<cam>:f<idx>`, `meta:<path>`); quality gates; audit scoring |
| `stage1.py`, `stage2.py` | 38, 95 | stage logic and gate/retry loops; ground truth is only in `stage2.gt_artifact()` |
| `mock.py` | 45 | canned Stage 1/2 responses for offline tests |
| `reasoning.py` | 51 | writes the API reasoning summaries to `stage*_reasoning.md` |
| `run_edge_slide.py` | 195 | Stage 1+2 runner (`--mock --dry-run --budget-usd --gt`), resumable |
| `run_stage1_blind.py` | 112 | optional: Stage 1 with all jaw-aperture facts removed from the metadata |
| `grade_edge_slide.py` + `rubrics/edge_slide.yaml` | 205, 62 | interactive hand grader (DRAFT rubric); `--llm-pregrade` is assist only |
| `prompts/stage1_system.md`, `stage2_system.md`, `stage2_retry.md` | | Stage 1/2 prompts |
| `prompts/stage3_system.md` | 22 | Stage 3 primitives system prompt (task, robot reference block, rules) |
| `prompts/stage3_code_system.md` | ~45 | Stage 3 code-mode system prompt (same robot reference + robot API + program rules) |
| `stage3/primitives.py` | 289 | `PrimitiveExecutor`: the 8 primitives on top of the env's delta-pose controller, `GripperRegister`, RPY helpers, gains and tolerances |
| `stage3/tools.py` | 139 | strict JSON tool schemas for the 8 primitives + pydantic validation (`parse_call`) |
| `stage3/agent.py` | ~380 | observation (state, camera calibration, optional `--obj` object info), prompt assembly, the episode loop for both modes, `MockToolClient`, `RenderError` |
| `stage3/memory.py` | ~68 | `ReactMemory` (last H steps; renders code programs in code mode) and `FrameHistory` |
| `stage3/plan.py` | 92 | PLAN block: `stage2` (from a Stage-2 run, optionally another seed's) / `oracle` / `none` |
| `stage3/code_policy.py` | ~350 | Code-as-Policies: AST sandbox, `CodeRunner` (robot API, limits, persistent namespace), `RUN_CODE` schema, `parse_code_call` |
| `stage3/scripted_program.py` | 91 | reference 3-turn program for the code-mode mock (needs `--obj`) |
| `stage3/primitive_report.py` | 310 | sim test of every primitive → `runs/stage3_primitive_report.md` |
| `run_stage3.py` | ~230 | Stage 3 runner: `--mode primitives|code`, `--obj`, `--plan-source`, `--stage2-seed`, `--K`, `--H`, `--mock`, `--dry-run`, `--render-prompts`; one subprocess per episode |
| `README.md` | 138 | commands + all protocol deviations |
| `TRANSFER.md` | | this file |

### Other code
| File | Role |
|---|---|
| `scripts/edge_slide_scripted_primitives.py` | scripted solver using only the 8 primitives (privileged plate pose). Feasibility proof, and the source of the primitives-mode `--mock` replay sequence |
| `tests/test_vlm_eval.py` (14 tests) | Stage 1/2: schemas, gates, mock pipeline, budget, leak checks |
| `tests/test_primitives.py` (14) | primitives in sim |
| `tests/test_stage3_agent.py` (23) | tool schemas, parsing, memory, plan injection, prompt slots, mock loop in sim, exits |
| `tests/test_code_policy.py` (33) | sandbox, runner limits, persistence, schemas, code prompt, mock loop in sim |

### Results: `runs/` (≈ 37 MB total)
| Path | Contents |
|---|---|
| `runs/gpt-6-astra/edge-slide/` (31 MB) | all real Astra runs (section 7) |
| `runs/stage3_scripted/`, `runs/stage3_scripted_holdout/` | scripted solver results (`results.json`, MP4s) |
| `runs/stage3_primitive_report.md`, `runs/stage3_primitive_report/` | primitive accuracy report |
| `runs/gpt-6-astra-MOCK/` (3.7 MB) | mock runs, no API (safe to drop) |

### Do NOT copy
- `~/openai_astra_key.txt` (the API key; recreate it on the new machine, `chmod 600`). This key was pasted in chat
  earlier, so revoke it.
- `CLAUDE.local.md` (contains live credentials).
- Anything pi0.5 / VLA / MolmoAct / DP related (`scripts/run_pi05_eval.py`, `scripts/run_vla_eval.py`,
  `imbench/policy/*`, `scripts/dp/*`, `data_synthetic/*`): separate projects.

---

## 3. Dependencies

### On IMBench (the new repo must vendor or pip-install IMBench)
The eval drives the real IMBench env, so IMBench must be importable as `imbench`:
- `imbench.make("ib-p1-edge-slide-v1", obs_mode="combo", max_episode_steps=N)` and `task.reset(seed=...)`,
  `task.step(a7)`, `task.close()`.
- `imbench.tasks.base_task.load_task_config("ib-p1-edge-slide-v1")["scene"]` (`table.height`, `lift_height`), used
  to build the task text.
- Internals used for privileged info: `task.env.sim` (MuJoCo), `task._find_body_id(sim, "cube")` (the plate body),
  geom `table_collision`, `task.camera_cfgs` (camera calibration).
- Observation keys used: `robot0_eef_pos`, `robot0_eef_quat` (xyzw), `robot0_gripper_qpos`, `force_torque`,
  `force_torque_world`, `robot0_tactile_left/right`, `object_pos`, `rgb_{exo_right_cam, exo_left_cam, ego_cam,
  gripper_cam}`; `info["task_score"]`.
- Config: `configs/tasks/prehensile/ib-p1-edge-slide-v1.yaml`. In the IMBench working tree it has uncommitted edits
  by others (DESCRIPTION is now "Single arm: Pick up the flat plate from the table"). Pin the IMBench commit and
  YAML you evaluate against.

**Import paths:** every module imports `imbench.vlm_eval...`. In the new repo either keep that package path (e.g.
copy into `imbench_vlm_eval/` and rewrite the imports), or install IMBench and keep `vlm_eval` as a separate package.
Rewrite with: `grep -rl "imbench.vlm_eval" | xargs sed -i 's/imbench\.vlm_eval/<newpkg>/g'`. Also fix
`scripts/edge_slide_scripted_primitives.py` (it inserts the repo root into `sys.path`).

**Paths hard-coded relative to the repo root:** `runs/...` (outputs), `runs/stage3_scripted/results.json`
(primitives `--mock`), `runs/gpt-6-astra/edge-slide/seed_X/sample_Y/` (Stage-2 plan for Stage 3). Run everything
from the repo root.

### Python environment (as tested)
Python 3.12.13; openai 3.42.0; pydantic 2.14.2; numpy 1.26.4; scipy 1.17.1; Pillow 13.0.0; imageio 2.38.1 (+ ffmpeg
plugin for MP4); PyYAML; pytest; robosuite 1.5.1; mujoco 3.2.6.
- Rendering: `MUJOCO_GL=egl` (headless).
- Tests: run with `-o addopts=""` (IMBench's pytest addopts use xdist, which isn't installed).
- API key: `OPENAI_API_KEY`, else the file `~/openai_astra_key.txt` (`client.KEY_FILE`).

---

## 4. How it works

### Stage 1 / Stage 2 (`run_edge_slide.py`)
1. `observation.build_bundle(seed)`: reset, hold still 0.5 s, save one 512 px frame per camera plus metadata
   (task description, camera descriptions, controller interface incl. the 8 cm max opening, sensor snapshot).
   **No object poses.**
2. Stage 1 (`max` effort, Structured Outputs → `ConstraintArtifact`): constraints with evidence refs. Gate checks
   the refs and the required fields; up to 3 attempts.
3. Stage 2 (`max` effort → `Stage2Plan`): sub-goals with physical outcomes, pre/post-conditions and constraint ids.
   Audited against `stage2.gt_artifact()`; up to 6 attempts, accept at audit mean ≥ 0.8.
4. Outputs per sample: `stage1*.json`, `stage2*.json`, `stage*_reasoning.md`, `attempts.jsonl`, `usage.json`.

### Stage 3 loop (`stage3/agent.py: run_episode`)
Per turn (max `K` = 24 turns), in a fresh subprocess per episode:
1. Build the observation: robot state (eef pose + rpy, gripper width/openness, F/T sensor + world force, tactile),
   camera calibration (position, view direction, up, fovy), elapsed steps; with `--obj` also plate pose/extents and
   table edges. Four current views (512 px, high detail) + the last `H` = 12 history batches (256 px, low detail).
2. Prompt = system prompt + PLAN block + frame history + current observation + working memory (last `H` turns:
   state before, call, result, reward change).
3. One API call with `tool_choice="required"`, `parallel_tool_calls=False`, effort `low`, `store=False`.
4. Validate the call. Invalid/failed → HOLD(20) and logged. Config errors (bad request, auth, quota exhausted) abort
   the episode.
5. Execute: primitives mode → `PrimitiveExecutor.run(name, params)`; code mode → `CodeRunner.run(code)`.
6. Stop on env success (`task_score ≥ 1`), `DONE`, termination, tick cap (10 000) or budget.

### The 8 primitives (`stage3/primitives.py`)
`MOVE_EEF(position_m, orientation_rpy_deg|null, max_steps)`, `CLOSE_GRIPPER`, `OPEN_GRIPPER`, `LIFT(dz_m)`,
`DESCEND(target_z_m)`, `ATOMIC_ACTION(action7, n_ticks ≤ 200)`, `HOLD(n_ticks ≤ 200)`, `DONE`.
- Built on the env's OSC_POSE delta controller (the YAML says `IK`, which is invalid, so the env falls back to OSC).
  100 Hz control. Each tick re-targets to achieved + a·(0.05 m / 0.5 rad), so actual motion is ≈ 2.8 mm and 0.63°
  per tick.
- MOVE_EEF: a P servo with velocity damping (KP 1.0, KD_POS 0.08 s, KD_ROT 0.1 s), rotation command capped at 0.3,
  slow-down within 2 cm. It converges when within 5 mm / 5° and nearly still (< 0.02 m/s, < 5°/s) for 3 ticks.
  Straight line, no obstacle avoidance, no stop on contact.
- HOLD and gripper actuation servo to a fixed pose (a zero delta lets the arm drift). The gripper keeps its last
  open/close state during moves.
- Returns: converged, steps, final pose, position/orientation error, peak force, contact flag (|ΔF| > 10 N or
  tactile > 0); for the gripper: finger width, `object_between_fingers` (width > empty-closed 0.0095 + 0.003 m),
  tactile.
- Accuracy (`primitive_report`, seeds 0–2): **138/138 checks pass**; commanded vs achieved 0.27 cm / 0.45° mean,
  0.48 cm / 1.05° worst.

### Code-as-Policies mode (`stage3/code_policy.py`)
- Tools: `RUN_CODE{reasoning, description, active_subgoal_id, subgoal_status, code}` and `DONE`.
- Robot API in the program: `move_eef`, `close_gripper`, `open_gripper`, `lift`, `descend`, `atomic_action`, `hold`,
  `done` (same validators as the tools); read-outs `get_state`, `eef_pos`, `eef_rpy`, `gripper_width`,
  `force_world`, `tactile`, `elapsed_steps`; `get_objects` **only with `--obj`**; `np`, `math`, `print`.
- Returned each turn: status (`ok`/`error`/`rejected`/`turn_limit`/`episode_stopped`/`done`), error with its line,
  stdout (≤ 4000 chars), primitive log (≤ 40 entries) with return values, the `result` variable, persistent names.
- Sandbox (best effort, not a security boundary): AST checks (no import, no `_` names/attrs, no
  exec/eval/open/getattr/type/…, no numpy file I/O attrs, no bare `except`/`except BaseException`, no
  global/nonlocal, no classes), restricted builtins. Per-turn limits: 3000 control steps, 120 primitive calls,
  600 s wall clock, 12 000 chars. Limits raise a `BaseException` subclass so model code can't catch them. The
  program stops when the env reports success.
- One namespace per episode, so helpers defined earlier see later globals; `print` is bound per episode and writes
  to the current turn's stdout.
- Prompt (`stage3_code_system.md`): same task text and robot reference as the primitives prompt. It states that
  programs are ordinary Python (loops, conditionals, helpers, closed-loop checks) and can also be a single
  primitive. This wording was added after the first code run, which used only 1–3-line programs; it has **not yet
  been tested on a full episode** (the run was cut short by the quota, see 7.4).

### Key sim facts (measured)
- World frame: robot base at (−0.41, 0, 0.912), identity rotation. Table top z 0.80, edges x, y = ±0.25. Plate
  rests at centre z ≈ 0.815.
- Success: plate centre z ≥ 0.95 (table + `lift_height` 0.15) with gripper contact.
- Gripper: max opening 8 cm, jaws close along the eef local y; empty close ≈ 0.0099 m. Down = rpy (180, 0, 0),
  extrinsic xyz in degrees; eef quat is xyzw.
- Friction: MuJoCo uses max(geom friction). Table μ = 1.0 and fingertips μ = 1.0, so pressing on the top face and
  dragging cannot slide the plate; **push the side face**.
- Reachability: side grasps work only at the y = ±0.25 edges, with rpy (side·90, 0, 0), i.e. (+90, 0, 0) at +y and
  (−90, 0, 0) at −y. The far edge x = +0.25 is not reachable for a side grasp, and the near edge x = −0.25 faces the
  robot base. **The prompt does not say this** (an open decision).
- Rendering bug: a second robosuite env in the same process can render all four cameras identically. Hence one
  subprocess per episode and the `RenderError` guard.

---

## 5. Commands (repo root)

```bash
export MUJOCO_GL=egl
# Stage 1/2
python -m imbench.vlm_eval.run_edge_slide --seeds 0 --samples 1 --dry-run
python -m imbench.vlm_eval.run_edge_slide --seeds 0 --samples 1 --budget-usd 10
python -m imbench.vlm_eval.grade_edge_slide
# Stage 3 checks (no API)
python -m imbench.vlm_eval.stage3.primitive_report --seeds 0 1 2
python scripts/edge_slide_scripted_primitives.py --seeds 0 1 2 3 4 5 6 7 8 9 --video-seeds 0
python -m imbench.vlm_eval.run_stage3 --seeds 0 --mock                                  # primitives replay
python -m imbench.vlm_eval.run_stage3 --seeds 0 1 2 --mock --mode code --obj --plan-source oracle
python -m imbench.vlm_eval.run_stage3 --seeds 0 --mode code --dry-run --plan-source stage2 --stage2-seed 0
# Stage 3 real (Astra, effort low)
python -m imbench.vlm_eval.run_stage3 --seeds 0 1 2 --plan-source stage2 --stage2-seed 0 --budget-usd 15 [--obj]
python -m imbench.vlm_eval.run_stage3 --mode code --seeds 0 --plan-source stage2 --stage2-seed 0 --budget-usd 8 [--obj]
# Tests
python -m pytest -o addopts="" tests/test_vlm_eval.py tests/test_primitives.py tests/test_stage3_agent.py tests/test_code_policy.py
```
Finished episodes (`summary.json` present) are skipped on re-run. Episode folders:
`runs/<model>/edge-slide/stage3[_code]/<plan>[-obj]/seed_<s>/ep_<e>/` with `system_prompt.md`, `tools.json`,
`steps.jsonl` (per turn: reasoning, call, validity, executed, result, state after, reward, usage, cost),
`step<k>_prompt.json` (images by path), `step<k>_raw.json`, `frames/`, `episode_<success|failure>.mp4`,
`summary.json`.

**Test status (2026-11-01):** 84/84 pass across the four test files (≈ 3 min, sim tests included).

---

## 6. Feasibility checks (no model)

| Check | Result |
|---|---|
| Primitive accuracy report, seeds 0–2 | 138/138 checks pass |
| Scripted solver (8 primitives only, privileged plate pose), seeds 0–9 | **10/10** |
| Scripted solver, held-out seeds 10–19 | **10/10** |
| Primitives-mode mock (scripted sequence through the agent loop) | success (test) |
| Code-mode mock (reference 3-turn program, `--obj`), seeds 0–2 | **3/3** |

Scripted strategy: closed jaws as a pusher, push the plate's side face through its centre toward the nearer ±y edge
in 1 cm closed-loop nudges until 3.5 cm overhang, lift, reorient at x = 0.05 to a sideways pose, open, straddle the
lip, close, lift 25 cm.

---

## 7. Astra results (all runs)

Model `gpt-6-astra`. Stage 1/2 at effort `max`; Stage 3 at effort `low`. Every Stage-3 episode used the **Stage-2
plan from seed 0** (9 sub-goals S1–S9: calibrate, inspect, stage, small pushes, create overhang, align on edge,
capture, proof lift, final lift), K = 24, H = 12. "Strategy" = whether it actually did the edge slide.

### 7.1 Stage 1 / Stage 2
| Seed | Result | Tokens (in / out / reasoning) | Cost |
|---|---|---|---|
| 0, sample 0 | Stage 1 gate passed 1st try; Stage 2 plan accepted 1st try, audit **0.95** | 8.5k / 38.4k / 32.2k | $2.00 |
| 1, sample 0 | Stage 1 gate passed; stopped before Stage 2 (decided to reuse seed 0's plan) | ~2.9k in | $0.93 |
| 2 | bundle only (frames, metadata), no API call | – | $0 |

Stage 1/2 grading (hand rubric) has **not been done** yet.

### 7.2 Stage 3, primitives mode
| # | Seed | `--obj` | Task text | Env result | What happened | Turns | Tokens (in / out / reas.) | Cost |
|---|---|---|---|---|---|---|---|---|
| P0 | 0 | (not recorded) | no height | invalid, deleted | API rejected effort `minimal`; 24 fallback HOLDs | 24 | 0 | $0 |
| P1 | 0 | yes | no height | **fail** (called DONE) | **Full edge slide** to −y, side grasp (plate between fingers), lifted only **4.8 cm** (z 0.863), then DONE. Skill worked; stopped low because the task text gave no height | 21 | 212k / 3.8k / 0.7k | $2.31 |
| P2 | 0 | no | no height | **fail** (called DONE) | Slid to +y overhang; grasp closed empty (1.0 cm); lifted the empty hand and **falsely called DONE** (plate z 0.815) | 24 | 245k / 4.8k / 1.1k | $2.69 |
| P3 | 1 | no | 15 cm | invalid, deleted | rendering bug: frozen views from turn 3 | – | – | $1.83 |
| P4 | 0 | no | 15 cm | fail (max turns) | Pushed toward the unreachable +x edge; approach stalled; empty close; correctly no DONE | 24 | 247k / 4.6k / 1.0k | $2.70 |
| P5 | 1 | no | 15 cm | fail (max turns) | Slid to −y overhang; approach stalled, plate tilted; empty close (0.98 cm); correctly no DONE | 24 | 247k / 6.3k / 2.7k | $2.79 |
| P6 | 2 | no | 15 cm | **env success** (turn 18) | **Not an edge slide.** Plate 14.4 × 8.2 cm, just over the 8 cm opening: jammed between the open fingers top-down, then closed and lifted to z 0.951 | 18 | 177k / 6.3k / 3.8k | $2.09 |

P1 and P2 are in `stage3_archive_v1_no_lift_height/`; P4–P6 in `stage3/stage2/`.

### 7.3 Stage 3, code mode
| # | Seed | `--obj` | Env result | What happened | Turns (primitives) | Tokens (in / out / reas.) | Cost |
|---|---|---|---|---|---|---|---|
| C1 | 0 | yes | **success** (turn 15) | **Real edge slide.** Rehearsed the sideways grasp pose at the −y edge before pushing; side-pushed the plate in shrinking steps (48 → 20 mm) while re-reading the pose; checked stability; turned sideways outside the edge (two stalls, recovered); closed (width 3.98 cm, plate between fingers); 4 cm proof lift; lifted 11.5 cm to z **0.951** | 15 (32) | 200k / 2.2k / 0.2k | $2.11 |
| C2 | 0 | no | invalid (overwritten) | First attempt: 4 real turns (test push toward the unreachable **+x** edge), then the OpenAI account ran out of credits (429); stopped. Folder overwritten by C3 | 7 (4 real) | – | $0.36 |
| C3 | 0 | no | **success** (turn 19) | **Real edge slide, no privileged info.** Judged edges from images (thought ±0.20 m; really ±0.25) and chose −y; pushed the side face in 5 → 3 cm steps (turns 3–7); held, lifted, turned sideways outside the edge (one stall, recovered by moving inward); 1st close **empty** (1.0 cm) → reopened, advanced 2 cm, captured (3.6 cm); proof lift **lost the plate** → diagnosed too little overhang, went back and pushed 4–6 cm more (turn 14); re-approached; another empty close → advanced, captured (3.98 cm, tactile 1.49); 2.5 cm proof lift held; lift → z **0.950** | 19 (46) | 253k / 3.3k / 0.7k | $2.70 |

C1: `stage3_code/stage2-obj/seed_0/ep_0/` (`episode_success.mp4`). Note: in C1 the programs were 1–3 lines
(motion + prints of objects/force/tactile), with no loops or helpers. So C1 shows richer feedback per turn, not
code-level control. The prompt was reworded afterwards (section 4) and C2 used the new wording.

### 7.4 Totals
| Item | Cost |
|---|---|
| Stage 1/2 | $2.93 |
| Stage 3 primitives, valid episodes (P1, P2, P4–P6) | $12.58 |
| Stage 3 code (C1, partial C2, C3) | $5.17 |
| Wasted (render bug P3) | $1.83 |
| **Total** | **≈ $22.51** |

Per-episode cost ≈ $2.1–2.8, almost all input tokens (≈ 180–250k per episode: images + history + memory). Output
and reasoning are small at effort `low`.

### 7.5 Scoreboard (valid Stage-3 episodes: P1, P2, P4, P5, P6, C1, C3)
- Env success: **3/7** (P6, C1, C3).
- Edge slide + lift ≥ 15 cm (intended strategy): **2/7** (C1 with `--obj`, C3 without), both code mode.
- Edge slide + any real lift: 3/7 (P1 4.8 cm, C1, C3).
- Plate pushed to overhang at a reachable edge: 5/7 (P1, P2, P5, C1, C3).
- False DONE: 2 (P1, P2, both under the old task text).
- Without `--obj`: primitives 0/3 intended-strategy (one wedge success, P6); code 1/1 (C3).

---

## 8. Lessons and gotchas
- **State the success height.** Without "≥ 15 cm (z ≥ 0.95 m)" the model stops at a few cm (P1).
- **Env success ≠ strategy.** Narrow plates (short side ≈ 8 cm) can be wedged top-down (P6). Check `steps.jsonl` /
  the MP4 before reporting; consider a strategy check (was overhang ≥ ~3 cm before the grasp?).
- **Empty grasps look like grasps to the model.** Close width ≈ 1 cm = empty. A tilted plate touching the pads gives
  small equal tactile readings (≈ 0.08) that the model read as a grip; real grips read ≈ 0.5 with width ≈ 3–4 cm.
- **The far edge (+x) is a trap.** Twice the model chose it (P4, C2). The prompt says nothing about reachability.
- **Rehearsing the grasp pose first** (C1, turn 3) is what avoided that trap.
- **24 turns is tight** for the long Stage-2 plan in primitives mode (P4, P5 ran out).
- **Effort names:** Astra accepts low/medium/high/xhigh/max; `minimal`/`none` are rejected (P0).
- **Quota:** a 429 "no credits" is now fatal (`QuotaExhausted`), not retried as a rate limit.
- **Rendering:** one subprocess per episode; never build two envs in one process.
- **Monitor scripts** that grep model text for keywords give false positives (e.g. "repeat").

---

## 9. Open items / next steps
1. ~~Re-run C2~~ done as C3 (success). Code mode still wrote short straight-line programs (no loops/helpers) even
   after the prompt rewording; the gain looks like multi-step turns + recovery, not code-level control.
2. **Controlled comparison** for C1: primitives mode, `--obj`, seed 0, current task text (≈ $2.5). Until then C1's
   success can't be attributed to code mode rather than the task text.
3. **Prompt decisions (not applied):** state which edges allow a side grasp; raise K to ~40 or use the oracle plan;
   whether to rule out the wedge grasp.
4. **Strategy-aware scoring:** report env success and intended-strategy success separately.
5. **Depth:** cameras are RGB only (`depth: false` in the YAML). The env supports `depth_<cam>` if enabled; option:
   add depth read-outs to code mode via an override in the eval code (don't edit the shared YAML).
6. **10-episode matrix** (stage2/oracle × with/without `--obj`, × modes): ≈ $2.5/episode, needs budget approval.
7. **Stage 1/2 grading** of seed 0 (interactive `grade_edge_slide`).
8. **Cleanup:** `runs/gpt-6-astra/edge-slide/seed_1/sample_0` (Stage 1 only, unused), `seed_2/` (bundle only),
   the partial C2 folder, `runs/gpt-6-astra-MOCK/`, the `_prompt_preview/` folder.
