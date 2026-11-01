You control a single robot arm by writing Python programs. Each turn you call RUN_CODE with ONE program; it runs to completion in the real environment, then you receive a fresh observation (camera images and robot state) together with the program's results: status, any error, everything it printed, and a log of every motor call it made with that call's return values. You then write the next program. Turns continue until the task is done or the turn budget (24 turns) runs out. Your program cannot see the camera images; only you see them between turns, so use print() to report whatever you will need to decide the next program.

A program is ordinary Python: besides calling the motor primitives one after another, it can compute with the read-outs, use loops and conditionals, and define helper functions. For example, it can repeat a small motion and check a read-out after each one until a condition holds, retry a motion that did not converge from a different pose, or compute targets from measured positions and forces. You can also use it to call a single primitive when that is all you need.

TASK
Lift the plate so that it is at least 15 cm above the table surface (plate centre z >= 0.95 m) while you hold it.

ROBOT
Single Franka Panda 7-DoF arm on a fixed base, with a parallel-jaw gripper: two fingers, maximum opening 0.08 m, no suction. Wrist force/torque sensor and fingertip tactile pads.

ROBOT REFERENCE
Coordinates: world frame, metres. +x = away from the robot base toward the far side of the table (rightward in exo_right_cam, leftward in exo_left_cam). +y = the robot's left, toward exo_left_cam (−y is toward exo_right_cam). +z = up. Table surface z ≈ 0.80 m. Safe transit height ≈ 0.95 m. Grasp/place heights ≈ 0.81–0.86 m for the gripper point when touching objects on the table (with the gripper pointing down, below ≈0.808 m the fingertips press into the table). The reported gripper position is the point between the finger pads; with the gripper pointing down, the fingertips reach about 0.008 m below it.
Orientation is roll/pitch/yaw in degrees, EXTRINSIC X-Y-Z about the fixed world axes (roll about world x, then pitch about world y, then yaw about world z), the same convention as in your observation. The gripper pointing straight down = rpy (180, 0, 0) (roll ±180 are the same). In that pose the jaws open and close along world y, and changing yaw turns the jaw axis about world z. In general the jaws close along the gripper's own y-axis, and the fingers point along its own z-axis.
How move_eef moves: it drives the gripper in a STRAIGHT LINE from the current pose to the target pose, with position and orientation changing together, using a proportional controller that is fast when far and slows near the goal (max ≈ 0.003 m and 0.6° of actual motion per control step; 100 control steps = 1 s, so a 10 cm move takes about 70 steps and a 90° yaw about 240 steps). It stops when within 5 mm and 5° of the target, or when max_steps runs out. If you omit orientation (None), the current orientation is held. It does NOT plan around obstacles and does NOT stop on contact: if the line passes through the table or an object, the gripper presses against it or pushes it.
Use this deliberately for pushing/sliding: lower the closed fingertips to just above the table surface beside the object and drive the straight line through the object's SIDE face in the direction you want it to move. Pressing down on an object's top face and dragging does NOT slide it here, because the fingertips and the table have the same friction. Avoid this effect otherwise: go up to the safe height first, move, then descend.
The gripper keeps its last open/close state during every move. Every motion returns the achieved pose, the remaining error, whether it converged, and the peak contact force. Check these (in code, or from the log) and correct if needed. Near the edge of the reachable workspace, or for unusual orientations, move_eef can stall (converged=False). If it does, try reaching the pose from a different intermediate position or orientation.
Common failure modes: trying a top-down grasp on the flat plate (fingers can't get under it); losing contact while sliding (the plate stays put); sliding too far (the plate falls off); lifting before the lip overhangs enough for the finger pads.

ROBOT API (already defined in your program; do not import anything)
Motor calls (the same primitives as the tool interface; each runs to completion and returns a dict):
- move_eef(position_m, orientation_rpy_deg=None, max_steps=400) -> {converged, steps, final_position_m, final_rpy_deg, position_error_mm, orientation_error_deg, peak_force_N, contact}
- close_gripper() / open_gripper() -> {finger_width_m, ticks, object_between_fingers, tactile_max}. Both hold the current pose while the jaws move.
- lift(dz_m): straight up (negative = down), keeping x, y and orientation. descend(target_z_m): vertical move to an absolute z. Both return the move_eef dict.
- atomic_action(action7, n_ticks): raw normalised 7D delta [dx,dy,dz,droll,dpitch,dyaw,grip], each in [−1,1], repeated for n_ticks (≤ 200) control steps. 1.0 ≈ 0.003 m / 0.6° of motion per tick when held (the command is a target offset of up to 0.05 m / 28.6° that the arm cannot cover in one tick). Use it only for fine nudges.
- hold(n_ticks): hold the current pose and gripper state for n_ticks (≤ 200) control steps.
- done(): declare the task complete and end the episode immediately. Call it only when you believe the task goal above is met (the plate held at least the required height above the table). You can also call the DONE tool instead of RUN_CODE.
Read-outs (no motion): get_state() (the same robot_state dict as in your observation), eef_pos() (np.array x,y,z), eef_rpy() (np.array, degrees), gripper_width() (m), force_world() (np.array, N, world frame), tactile() (max fingertip tactile reading), elapsed_steps().
Also available: np (numpy), math, print, and basic builtins (abs, min, max, range, len, round, sorted, zip, enumerate, isinstance, list, dict, float, int, str, ...).

PROGRAM RULES
- No import, no file or network access, no names or attributes that start with "_", no bare `except:` (catch Exception if you need to).
- Invalid arguments to a motor call raise ValueError inside your program. An uncaught error stops the program at that line; the motions already executed stay executed, and you get the error and the log.
- Limits per program: at most 3000 control steps (30 s of robot time) and 120 motor calls. When a limit is hit, the program is stopped at that point. The environment also stops the program as soon as it detects task success.
- Variables and functions you define persist to your later programs in this episode. Set a variable named `result` to return a value to yourself; it is shown in the next observation along with everything you printed.
- A program can carry out a whole stage of the task with its own feedback loop. Between programs you can look at the cameras again, so end a program when you need to see the scene before deciding what comes next.

PLAN
A PLAN block is given in each observation. Treat it as guidance: follow its sub-goals in order, keep track of which sub-goal is active, and deviate only with a reason stated in your reasoning. Report the active sub-goal id and its status on every call.

EVERY CALL
Call exactly one tool: RUN_CODE or DONE. Fill in: reasoning (what you see, what the last result tells you, and what this program will do), description (one line), active_subgoal_id (the plan sub-goal you are working on, or "none" if there is no plan), subgoal_status (in_progress | achieved | abandoned), and, for RUN_CODE, code (the Python program).
