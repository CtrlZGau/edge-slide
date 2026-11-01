"""Edge Slide and Grasp — push a flat object to the table edge, then grasp and lift.

A flat object (book/card) lies flush on the table. A parallel-jaw gripper
cannot grasp it from above because there is no vertical clearance. The robot
must slide the object to the table edge so that part of it overhangs, creating
a graspable lip, then grasp and lift it.

Default reward is sparse: 1.0 on success, 0.0 otherwise.
Pass ``reward_fn`` to override with a custom (e.g. dense) reward.

Usage
-----
    import imbench as ib

    # Sparse reward (default).
    task = ib.make("ib-p1-edge-slide-v1")

    # Custom dense reward — full access to sim internals:
    def my_reward(task, raw_obs):
        sim = task.sim                          # MuJoCo sim
        obj_pos = raw_obs["cube_pos"]           # raw robosuite obs
        eef_pos = raw_obs["robot0_eef_pos"]
        table_h = task._table_height            # task config
        grasped = task._gripper_contacts_object()  # task helpers
        return -np.linalg.norm(eef_pos - obj_pos) + 5.0 * grasped

    task = ib.make("ib-p1-edge-slide-v1", reward_fn=my_reward)
"""

import numpy as np
from robosuite.utils.placement_samplers import UniformRandomSampler
from imbench.tasks.base_task import BaseTask
import imbench.envs.edge_slide_env  # noqa: F401  (registers EdgeSlideEnv with robosuite)


class EdgeSlideV1(BaseTask):
    TASK_ID = "ib-p1-edge-slide-v1"
    DESCRIPTION = "Single arm: Pick up the flat plate from the table"
    def __init__(self, reward_fn=None, **kwargs):
        self._reward_fn = reward_fn
        # Per-episode state — reset in reset().
        self._rng = np.random.RandomState(0)
        self._book_fell_off = False
        super().__init__(**kwargs)

    # ── public accessors for reward authors ─────────────────────────────

    @property
    def sim(self):
        """The raw MuJoCo sim object — full access to model/data."""
        return self._env.sim

    @property
    def env(self):
        """The underlying robosuite environment."""
        return self._env

    # ── step override: sparse default reward + optional custom reward ───

    def step(self, action: np.ndarray):
        raw_obs, _robosuite_reward, done, info = self._env.step(action)
        self._step_count += 1
        self._inject_ft_sensor(raw_obs)
        self._inject_tactile_obs(raw_obs)
        obs = self._get_obs(raw_obs)
        task_score = self._compute_task_score(raw_obs)
        terminated = self._check_termination(raw_obs)
        truncated = self._step_count >= self.MAX_EPISODE_STEPS

        # Reward: custom fn if provided, else sparse (1.0 success / 0.0).
        if self._reward_fn is not None:
            reward = float(self._reward_fn(self, raw_obs))
        else:
            reward = task_score

        info.update({"task_score": task_score, "step": self._step_count})
        return obs, reward, terminated, truncated, info

    # ── environment construction ────────────────────────────────────────

    def _parse_task_config(self):
        """Read task-specific params from the YAML config (self.cfg)."""
        scene = self.cfg.get("scene", {})
        table_cfg = scene.get("table", {})
        obj_cfg = scene.get("object", {})
        place_cfg = self.cfg.get("placement", {})
        rand_cfg = self.cfg.get("randomization", {})

        self._table_height = table_cfg.get("height", 0.8)
        self._table_full_size = tuple(table_cfg.get("full_size", [0.5, 0.5, 0.05]))
        _offset = table_cfg.get("offset")
        self._table_offset = tuple(_offset) if _offset else (0, 0, self._table_height)
        self._robot_offset = scene.get("robot_offset")  # list of [[x,y,z]] or None
        self._lift_threshold = self._table_height + scene.get("lift_height", 0.05)

        # Object base dimensions (used when randomization range is absent).
        self._book_half_length = obj_cfg.get("half_length", 0.075)
        self._book_half_width = obj_cfg.get("half_width", 0.05)
        self._book_half_height = obj_cfg.get("half_height", 0.0015)
        self._book_mass = obj_cfg.get("mass", 0.1)
        self._book_friction = obj_cfg.get("friction", 0.4)
        self._book_color = obj_cfg.get("color", [0.5, 0.0, 0.0, 1.0])
        self._table_color = scene.get("table_color", [1.0, 1.0, 1.0, 1.0])

        # Robot base placement (None → robosuite default).
        _roff = scene.get("robot_offset")
        _rori = scene.get("robot_ori")
        self._robot_offset = tuple(_roff) if _roff else None
        self._robot_ori = tuple(_rori) if _rori else None

        # Visual randomization flags.
        self._rand_object_color = rand_cfg.get("object_color", {}).get("enabled", False)
        self._rand_table_color = rand_cfg.get("table_color", {}).get("enabled", False)
        light_cfg = rand_cfg.get("lighting", {})
        self._rand_light_position = light_cfg.get("position", False)
        self._rand_light_color = light_cfg.get("color", False)

        self._place_x_range = place_cfg.get("x_range", 0.12)
        self._place_y_range = place_cfg.get("y_range", 0.12)
        _center = place_cfg.get("center")
        self._place_center = tuple(_center) if _center else (0.0, 0.0)

        # Optional randomization — None when the key is absent in YAML.
        len_rand = rand_cfg.get("object_half_length", {}).get("range")
        wid_rand = rand_cfg.get("object_half_width", {}).get("range")
        fric_rand = rand_cfg.get("object_friction", {}).get("range")
        self._book_length_range = tuple(len_rand) if len_rand else None
        self._book_width_range = tuple(wid_rand) if wid_rand else None
        self._book_friction_range = tuple(fric_rand) if fric_rand else None

    def _build_env(self, **kwargs):
        # Parse config here — self.cfg is available but super().__init__ isn't done yet.
        self._parse_task_config()

        # If color randomization is enabled, pick a random initial color so
        # the compiled MuJoCo model never shows the static default red.
        if self._rand_object_color:
            self._book_color = list(self._random_object_color())

        # Deterministic placement — we randomize position manually in reset().
        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=None,
            x_range=[0.0, 0.0],
            y_range=[0.0, 0.0],
            rotation=0.0,
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            reference_pos=np.array([0.0, 0.0, self._table_height]),
            z_offset=0.002,
        )

        use_cameras = self.obs_mode in ("rgb", "combo") and self.has_cameras

        # Use the upper bound of each randomisation range (or the fixed
        # value) so that MuJoCo compiles collision structures large enough
        # for every possible book size.
        max_half_len = (self._book_length_range[1]
                        if self._book_length_range else self._book_half_length)
        max_half_wid = (self._book_width_range[1]
                        if self._book_width_range else self._book_half_width)
        max_friction = (self._book_friction_range[1]
                        if self._book_friction_range else self._book_friction)

        env_kwargs = dict(
            book_size=(max_half_len, max_half_wid, self._book_half_height),
            book_color=self._book_color,
            book_friction=(max_friction, 0.005, 0.0001),
            robot_offset=self._robot_offset,
            robot_ori=self._robot_ori,
            robots="Panda",
            table_full_size=self._table_full_size,
            table_offset=self._table_offset,
            control_freq=self.control_freq,
            controller_configs=self.controller_configs,
            placement_initializer=placement_initializer,
            has_renderer=(self.render_mode == "human"),
            render_camera=self.viewer_camera,
            renderer_config={"cam_config": self._viewer_cam_config} if self.render_mode == "human" else None,
            has_offscreen_renderer=use_cameras,
            use_camera_obs=use_cameras,
            camera_names=self._camera_names() if use_cameras else None,
            camera_heights=self._camera_heights() if use_cameras else 84,
            camera_widths=self._camera_widths() if use_cameras else 84,
            reward_shaping=(self.reward_type == "dense"),
            horizon=self.MAX_EPISODE_STEPS,
        )
        if self._robot_offset is not None:
            env_kwargs["robot_offset"] = self._robot_offset
        env = self._make_robosuite_env("EdgeSlideEnv", **env_kwargs)
        return env

    # ── reset: randomize book size / position / visuals ──────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self._rng = np.random.RandomState(self.seed)
        self._book_fell_off = False

        sim = self._env.sim

        # --- book geometry: use randomization range if set, else fixed value ---
        half_len = (self._rng.uniform(*self._book_length_range)
                    if self._book_length_range else self._book_half_length)
        half_wid = (self._rng.uniform(*self._book_width_range)
                    if self._book_width_range else self._book_half_width)
        half_ht = self._book_half_height
        friction = (self._rng.uniform(*self._book_friction_range)
                    if self._book_friction_range else self._book_friction)

        # Apply randomised dimensions.  The model was compiled at the upper
        # bound of each range (see _build_env), so shrinking here is always
        # safe — the compiled geom_rbound is large enough for any value
        # within the range.
        cube_body_id = self._find_body_id(sim, "cube")
        if cube_body_id is not None:
            for i in range(sim.model.ngeom):
                if sim.model.geom_bodyid[i] == cube_body_id:
                    sim.model.geom_size[i] = [half_len, half_wid, half_ht]
                    sim.model.geom_friction[i] = [friction, 0.005, 0.0001]

        # Set mass on the cube body.
        if cube_body_id is not None:
            sim.model.body_mass[cube_body_id] = self._book_mass

        # --- randomize position on the table surface ---
        x = self._place_center[0] + self._rng.uniform(-self._place_x_range, self._place_x_range)
        y = self._place_center[1] + self._rng.uniform(-self._place_y_range, self._place_y_range)
        yaw = self._rng.uniform(0, 2 * np.pi)
        z = self._table_height + half_ht + 0.001

        self._teleport_cube(sim, x, y, z, yaw)

        # --- visual randomization ---
        self._randomize_visuals(sim)

        sim.forward()

        # Re-observe after modifications.  force_update=True is required
        # because robosuite caches observables per timestep — without it
        # the cameras return stale images from before visual randomization.
        self._camera_cache.clear()
        raw_obs = self._get_sim_observations(force_update=True)

        # Patch cube position from sim state — robosuite Observable may
        # return a stale cached value after qpos-level teleport.
        cube_body_id = self._find_body_id(sim, "cube")
        if cube_body_id is not None:
            raw_obs["cube_pos"] = np.array(sim.data.body_xpos[cube_body_id])

        self._inject_ft_sensor(raw_obs)
        self._inject_tactile_obs(raw_obs)
        obs = self._get_obs(raw_obs)
        info = {"task_score": 0.0, "step": 0}
        return obs, info

    # ── observation ─────────────────────────────────────────────────────

    def _get_obs(self, raw_obs):
        obs = self._build_obs_from_config(raw_obs)

        if self.object_state_enabled:
            if "cube_pos" in raw_obs:
                obs["object_pos"] = raw_obs["cube_pos"]
            if "cube_quat" in raw_obs:
                obs["object_quat"] = raw_obs["cube_quat"]

        return obs

    # ── task score ──────────────────────────────────────────────────────

    def _compute_task_score(self, raw_obs):
        if "cube_pos" not in raw_obs:
            return 0.0

        obj_z = raw_obs["cube_pos"][2]

        # Object fell off the table.
        if obj_z < self._table_height - 0.05:
            self._book_fell_off = True
            return 0.0

        # Already failed earlier in this episode.
        if self._book_fell_off:
            return 0.0

        # Full success: lifted above threshold with gripper contact.
        if obj_z >= self._lift_threshold and self._gripper_contacts_object():
            return 1.0

        return 0.0

    # ── termination ─────────────────────────────────────────────────────

    def _check_termination(self, raw_obs):
        if "cube_pos" not in raw_obs:
            return False

        # Terminate if book fell off table.
        if self._book_fell_off:
            return True

        obj_z = raw_obs["cube_pos"][2]
        if obj_z < self._table_height - 0.05:
            self._book_fell_off = True
            return True

        # Terminate on full success.
        if obj_z >= self._lift_threshold and self._gripper_contacts_object():
            return True

        return False

    # ── helpers ──────────────────────────────────────────────────────────

    def _gripper_contacts_object(self):
        """Return True if any gripper geom is in contact with a cube geom."""
        sim = self._env.sim
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            g1 = sim.model.geom_id2name(contact.geom1)
            g2 = sim.model.geom_id2name(contact.geom2)
            if g1 is None or g2 is None:
                continue
            g1_grip = "gripper" in g1 or "finger" in g1 or "taxel" in g1
            g2_grip = "gripper" in g2 or "finger" in g2 or "taxel" in g2
            g1_cube = "cube" in g1
            g2_cube = "cube" in g2
            if (g1_grip and g2_cube) or (g2_grip and g1_cube):
                return True
        return False

    # ── visual randomization ────────────────────────────────────────────

    def _randomize_visuals(self, sim):
        """Apply visual domain randomization (colors + lighting) to the sim."""

        # ── Object color ────────────────────────────────────────────────
        obj_rgba = self._random_object_color() if self._rand_object_color else np.array(self._book_color)
        cube_body_id = self._find_body_id(sim, "cube")
        if cube_body_id is not None:
            for i in range(sim.model.ngeom):
                if sim.model.geom_bodyid[i] == cube_body_id:
                    # Strip the material so geom_rgba is the solid color
                    # (otherwise the redwood texture mutes the color).
                    sim.model.geom_matid[i] = -1
                    sim.model.geom_rgba[i] = obj_rgba

        # ── Table color ─────────────────────────────────────────────────
        table_rgba = self._random_table_color() if self._rand_table_color else np.array(self._table_color)
        for i in range(sim.model.ngeom):
            name = sim.model.geom_id2name(i)
            if name and name == "table_visual":
                sim.model.geom_rgba[i] = table_rgba
                mat_id = sim.model.geom_matid[i]
                if mat_id >= 0:
                    sim.model.mat_rgba[mat_id] = table_rgba

        # ── Lighting randomization ──────────────────────────────────────
        for i in range(sim.model.nlight):
            if self._rand_light_position:
                sim.model.light_pos[i] = [
                    self._rng.uniform(0.0, 2.0),
                    self._rng.uniform(-1.0, 2.0),
                    self._rng.uniform(1.0, 2.5),
                ]
            if self._rand_light_color:
                # Near-white warm-to-cool lighting; avoid strongly colored
                # lights (e.g. red, green, blue).
                base = self._rng.uniform(0.6, 1.0)
                jitter = self._rng.uniform(-0.08, 0.08, size=3)
                diffuse = np.clip(base + jitter, 0.4, 1.0)
                sim.model.light_diffuse[i] = diffuse
                amb_base = self._rng.uniform(0.05, 0.25)
                amb_jitter = self._rng.uniform(-0.04, 0.04, size=3)
                sim.model.light_ambient[i] = np.clip(
                    amb_base + amb_jitter, 0.0, 0.4
                )

    # ── color helpers (realistic palette) ───────────────────────────

    def _random_object_color(self):
        """Sample a plausible object color — muted, avoids neon extremes."""
        rgb = self._rng.uniform(0.15, 0.85, size=3)
        gray = rgb.mean()
        rgb = 0.5 * rgb + 0.5 * gray
        rgb = np.clip(rgb, 0.1, 0.9)
        return np.array([*rgb, 1.0])

    def _random_table_color(self):
        """Sample a realistic table-top color (neutrals: grays, off-whites, light wood)."""
        base = self._rng.uniform(0.45, 0.95)
        jitter = self._rng.uniform(-0.06, 0.06, size=3)
        jitter[0] += self._rng.uniform(0.0, 0.04)  # slight warm tint
        rgb = np.clip(base + jitter, 0.3, 1.0)
        return np.array([*rgb, 1.0])

    @staticmethod
    def _find_body_id(sim, keyword):
        """Return the first body id whose name contains *keyword*, or None."""
        for i in range(sim.model.nbody):
            name = sim.model.body_id2name(i)
            if name and keyword in name:
                return i
        return None

    def _teleport_cube(self, sim, x, y, z, yaw):
        """Move the cube to the given position and yaw angle."""
        joint_name = None
        for name in sim.model.joint_names:
            if "cube" in name:
                joint_name = name
                break
        if joint_name is None:
            return

        addr = sim.model.get_joint_qpos_addr(joint_name)
        if isinstance(addr, tuple):
            addr = addr[0]

        qw = np.cos(yaw / 2)
        qz = np.sin(yaw / 2)
        sim.data.qpos[addr:addr + 7] = [x, y, z, qw, 0.0, 0.0, qz]
        # Zero out velocity.
        vel_addr = sim.model.get_joint_qvel_addr(joint_name)
        if isinstance(vel_addr, tuple):
            vel_addr = vel_addr[0]
        sim.data.qvel[vel_addr:vel_addr + 6] = 0.0
