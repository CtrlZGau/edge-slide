"""Base class for single-arm tasks using SingleArmTableEnv.

Provides shared config parsing, obs building, visual randomization,
and termination helpers that are common across most single-arm tasks.
"""

import numpy as np

import imbench.envs  # noqa: F401 — register envs
from imbench.tasks.base_task import BaseTask


class SingleArmBase(BaseTask):
    """Shared base for single-arm prehensile and puzzle tasks.

    Subclasses must set ``TASK_ID`` and may override:
    - ``_parse_task_config()`` — to read task-specific YAML keys
    - ``_make_objects()`` — return list of object spec dicts
    - ``_make_static_objects()`` — return list of static object spec dicts
    - ``_compute_task_score()`` / ``_check_termination()``
    """

    TASK_ID: str = ""
    DESCRIPTION: str = ""

    def __init__(self, reward_fn=None, **kwargs):
        self._reward_fn = reward_fn
        self._rng = np.random.RandomState(0)
        self._score_memo_key: object | None = None
        self._score_memo_val: float | None = None
        self._in_step_context: bool = False
        super().__init__(**kwargs)
        # Wrap _compute_task_score with a per-tick identity cache so that when
        # _check_termination calls _compute_task_score on the same raw_obs dict,
        # it reuses the already-computed value and avoids advancing per-tick
        # side-effect counters (e.g. _stable_count) a second time.

        _impl = self._compute_task_score
        def _memoised(raw_obs, _self=self, _impl=_impl):
            if _self._in_step_context and _self._score_memo_key is raw_obs:
                return _self._score_memo_val
            val = _impl(raw_obs)
            if _self._in_step_context:
                _self._score_memo_key = raw_obs
                _self._score_memo_val = val
            return val
        self._compute_task_score = _memoised

    # ── Public accessors ────────────────────────────────────────────

    @property
    def sim(self):
        return self._env.sim

    @property
    def env(self):
        return self._env

    # ── Step override with optional custom reward ───────────────────

    def step(self, action: np.ndarray):
        raw_obs, _rs_reward, done, info = self._env.step(action)
        self._step_count += 1
        self._inject_ft_sensor(raw_obs)
        self._inject_tactile_obs(raw_obs)
        obs = self._get_obs(raw_obs)

        # Clear the per-tick score cache so the first _compute_task_score
        # call below computes fresh; subsequent calls within this step
        # (e.g. from _check_termination) reuse the cached value.
        self._score_memo_key = None
        self._score_memo_val = None
        self._in_step_context = True
        try:
            task_score = self._compute_task_score(raw_obs)
            terminated = bool(self._check_termination(raw_obs))
        finally:
            self._in_step_context = False
        truncated = bool(self._step_count >= self.MAX_EPISODE_STEPS)

        if self._reward_fn is not None:
            reward = float(self._reward_fn(self, raw_obs))
        else:
            reward = task_score

        info.update({"task_score": task_score, "step": self._step_count})
        return obs, reward, terminated, truncated, info

    # ── Config parsing ──────────────────────────────────────────────

    def _parse_common_config(self):
        """Parse scene, table, randomization keys shared by all single-arm tasks."""
        scene = self.cfg.get("scene", {})
        table_cfg = scene.get("table", {})
        rand_cfg = self.cfg.get("randomization", {})

        self._table_height = table_cfg.get("height", 0.8)
        self._table_full_size = tuple(table_cfg.get("full_size", [0.5, 0.5, 0.05]))
        _offset = table_cfg.get("offset")
        self._table_offset = tuple(_offset) if _offset else (0, 0, self._table_height)
        self._robot_offset = scene.get("robot_offset")

        # Visual randomization
        self._rand_object_color = rand_cfg.get("object_color", {}).get("enabled", False)
        self._rand_table_color = rand_cfg.get("table_color", {}).get("enabled", False)
        light_cfg = rand_cfg.get("lighting", {})
        self._rand_light_position = light_cfg.get("position", False)
        self._rand_light_color = light_cfg.get("color", False)

        # Task conditions (optional)
        cond_cfg = self.cfg.get("task_conditions", {})
        success_cfg = cond_cfg.get("success", {})
        failure_cfg = cond_cfg.get("failure", {})
        self._success_score_threshold = success_cfg.get("score_threshold", 1.0)
        self._failure_floor_threshold = failure_cfg.get("floor_z_threshold", 0.05)

    def _parse_task_config(self):
        """Override in subclass to parse task-specific config keys."""
        pass

    # ── Object factory — subclasses override these ──────────────────

    def _make_objects(self):
        """Return list of object spec dicts for dynamic objects."""
        return []

    def _make_static_objects(self):
        """Return list of object spec dicts for static objects."""
        return []

    # ── Environment construction ────────────────────────────────────

    def _build_env(self, **kwargs):
        self._parse_common_config()
        self._parse_task_config()

        use_cameras = self.obs_mode in ("rgb", "combo") and self.has_cameras

        objects = self._make_objects()
        static_objects = self._make_static_objects()

        placement_cfg = self.cfg.get("placement", {})
        x_range = placement_cfg.get("x_range", 0.0)
        y_range = placement_cfg.get("y_range", 0.0)
        if isinstance(x_range, (int, float)):
            x_range = [-x_range, x_range]
        if isinstance(y_range, (int, float)):
            y_range = [-y_range, y_range]

        env_kwargs = dict(
            objects=objects,
            static_objects=static_objects,
            robots="Panda",
            table_full_size=self._table_full_size,
            table_offset=self._table_offset,
            placement_config={
                "x_range": x_range,
                "y_range": y_range,
                "z_offset": placement_cfg.get("z_offset", 0.01),
                "reference_pos": [0.0, 0.0, self._table_height],
                "ensure_valid_placement": placement_cfg.get("ensure_valid_placement", True),
            },
            control_freq=self.control_freq,
            controller_configs=self.controller_configs,
            has_renderer=(self.render_mode == "human"),
            render_camera=None,
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

        return self._make_robosuite_env("SingleArmTableEnv", **env_kwargs)

    # ── Obs ─────────────────────────────────────────────────────────

    def _get_obs(self, raw_obs):
        obs = self._build_obs_from_config(raw_obs)
        # Add object positions if available
        if "cube_pos" in raw_obs:
            obs["object_pos"] = raw_obs["cube_pos"]
        if "cube_quat" in raw_obs:
            obs["object_quat"] = raw_obs["cube_quat"]
        return obs

    # ── Reset with visual randomization ─────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Only re-seed _rng when an explicit seed is provided;
        # otherwise keep the existing RNG so draws differ across episodes.
        if seed is not None:
            self._rng = np.random.RandomState(self.seed)

        sim = self._env.sim
        # Constrained light randomization
        if self._rand_light_color:
            for i in range(sim.model.nlight):
                sim.model.light_diffuse[i] = self._rng.uniform(0.6, 1.0, size=3)
                sim.model.light_specular[i] = sim.model.light_diffuse[i] * 0.5
        if self._rand_light_position:
            for i in range(sim.model.nlight):
                sim.model.light_pos[i][:2] += self._rng.uniform(-0.3, 0.3, size=2)
                sim.model.light_pos[i][2] = self._rng.uniform(1.0, 2.0)

        return obs, info

    # ── Visual randomization helpers ────────────────────────────────

    def _random_object_color(self):
        rgb = self._rng.uniform(0.1, 0.9, size=3)
        return np.array([*rgb, 1.0])

    def _random_table_color(self):
        rgb = self._rng.uniform(0.3, 1.0, size=3)
        return np.array([*rgb, 1.0])

    # ── Termination helper ──────────────────────────────────────────

    def _object_fell_off_table(self, raw_obs, pos_key="cube_pos"):
        """Check if object z is below floor threshold."""
        if pos_key in raw_obs:
            return raw_obs[pos_key][2] < self._failure_floor_threshold
        return False

    # ── Object positioning helpers ──────────────────────────────────

    def _place_object(self, name, pos, quat=None):
        """Reposition a dynamic (free-joint) object after sampler placement.

        Parameters
        ----------
        name : str
            Object name (e.g. ``"cube"``, ``"domino_0"``).
        pos : array-like
            ``[x, y, z]`` world position.
        quat : array-like, optional
            ``[w, x, y, z]`` MuJoCo quaternion.  Defaults to ``[1, 0, 0, 0]``.
        """
        if quat is None:
            quat = [1, 0, 0, 0]
        obj = self._env.get_object(name)
        if obj is not None and obj.joints:
            self._env.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([np.array(pos, dtype=float),
                                np.array(quat, dtype=float)]),
            )

    def _place_static(self, name, pos, quat=None):
        """Reposition a static (welded) object.

        Parameters
        ----------
        name : str
            Object name (e.g. ``"goal_marker"``, ``"occluder"``).
        pos : array-like
            ``[x, y, z]`` world position.
        quat : array-like, optional
            ``[w, x, y, z]`` MuJoCo quaternion.
        """
        sim = self._env.sim
        body_id = sim.model.body_name2id(f"{name}_main")
        sim.model.body_pos[body_id] = np.array(pos, dtype=float)
        if quat is not None:
            sim.model.body_quat[body_id] = np.array(quat, dtype=float)

    def _set_object_velocity(self, name, linvel, angvel=None):
        """Set velocity for a dynamic object.

        Parameters
        ----------
        name : str
            Object name.
        linvel : array-like
            ``[vx, vy, vz]`` linear velocity.
        angvel : array-like, optional
            ``[wx, wy, wz]`` angular velocity. Defaults to zero.
        """
        sim = self._env.sim
        # Composite objects (e.g. hex_prism, cone_frustum) register their root
        # body as "<name>_root"; primitives use "<name>_main".
        body_id = None
        for suffix in ("_main", "_root", ""):
            try:
                body_id = sim.model.body_name2id(name + suffix)
                break
            except Exception:
                continue
        if body_id is None:
            raise ValueError(f"No body found for object '{name}'")
        jnt_addr = sim.model.body_jntadr[body_id]
        if jnt_addr >= 0:
            dof_addr = sim.model.jnt_dofadr[jnt_addr]
            sim.data.qvel[dof_addr:dof_addr + 3] = np.array(linvel, dtype=float)
            if angvel is not None:
                sim.data.qvel[dof_addr + 3:dof_addr + 6] = np.array(angvel, dtype=float)

    def _hotswap_env(self):
        """Close and rebuild the robosuite env when dynamic object count changes.

        Call this from a task's reset() *before* super().reset() whenever the
        number of objects returned by _make_objects() needs to change.  After
        rebuild the MuJoCo model matches the new count and all custom cameras
        are re-injected so YAML camera configs remain valid.
        """
        try:
            self._env.close()
        except Exception:
            pass
        self._env = self._build_env()
        self._tactile_robot_indices = getattr(self._env, "_tactile_robot_indices", [])
        if self._custom_cameras:
            self._inject_custom_cameras()
        self._camera_cache.clear()

    def _gripper_contacts_object(self, obj_name="cube"):
        """Check if gripper is in contact with the named object."""
        obj = self._env.get_object(obj_name)
        geoms = obj.contact_geoms if obj is not None else obj_name
        return self._check_grasp(
            gripper=self._env.robots[0].gripper,
            object_geoms=geoms,
        )
