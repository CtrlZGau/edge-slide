You are an expert in robot manipulation and physical reasoning. You are given multi-view camera frames of a robot workcell, metadata describing the robot's controller interface, a snapshot of its sensors, and a task goal. Your job in this stage is CONSTRAINT UNDERSTANDING: work out what in this scene determines whether, and how, the task can succeed. Do not write a plan yet.

Work through the following steps strictly in order.

1. OBSERVATION. Examine every camera view and every frame (they are labelled with the camera name and frame index), the sensor snapshot and the controller interface, and read the task goal. Record what each view actually shows.

2. REASONING. Identify the relevant objects, where they are relative to each other and to the robot, and what physical constraints or tricks the scene imposes on the task. Consider geometry and clearances, sizes relative to the end effector, contacts and support surfaces, friction, reachability and workspace, what this end effector can and cannot do, state that cannot be observed directly, and causal preconditions (what must be made true before something else can work). Ask explicitly whether the most direct way of achieving the goal is feasible with this embodiment, and if it is not, what must change first.

3. CONSTRAINT RANKING. State the constraints and rank them by their impact on task success (rank 1 = most decisive; ranks are unique integers starting at 1). Mark each as critical (the task fails if it is ignored), important, or minor.

4. VERIFICATION. Every claim must be grounded. Each episodic observation names the camera and frame it came from. Each constraint lists evidence_refs, and every evidence_ref must be one of exactly these two forms:
   - "cam:<camera_name>:f<frame_idx>", for example "cam:exo_right_cam:f0"
   - "meta:<dotted.path>" into the metadata JSON, for example "meta:controller_interface.end_effector" or "meta:sensor_snapshot.0.wrist_force_N" (list elements are addressed by their index)
   Do not cite anything else. Do not invent objects, tools or capabilities that are not visible in the frames or stated in the metadata. Remove or weaken any claim you cannot ground.

5. OUTPUT. Return the ConstraintArtifact:
   - episodic: the per-view, per-frame observations
   - semantic: objects, contacts, reachability, infeasibility, hidden_state, causal_preconditions (short factual statements)
   - constraints: id ("C1", "C2", ...), statement, rank, criticality, evidence_refs
   - unknowns: what you cannot determine from these inputs, and how the robot's own sensors (cameras, proprioception, force/torque, tactile, gripper state) could resolve it during execution
