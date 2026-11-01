from __future__ import annotations

import logging
import os
import warnings
from abc import abstractmethod
from pathlib import Path
import gymnasium
from gymnasium import spaces
import numpy as np
import yaml
import json
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import REGISTERED_ENVS

logger = logging.getLogger(__name__)

from imbench.utils.tactile_setup import (
    make_tactile_env_class,
    get_tactile_readings as _tactile_readings,
)


CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "tasks"


def load_task_config(task_id: str, config_file: str | None = None, overrides: dict | None = None) -> dict:
    """Load YAML config for a task, with optional dict overrides.

    Parameters
    ----------
    task_id : str
        Task identifier (used as filename when *config_file* is ``None``).
    config_file : str, optional
        Explicit config filename (without ``.yaml``).  When set, the config
        is loaded from ``configs/tasks/{config_file}.yaml`` instead of
        ``{task_id}.yaml``.  Useful for task families that share one config.
    overrides : dict, optional
        Recursive dict merge applied on top of the loaded YAML.
    """
    fname = config_file or task_id
    cfg_path = CONFIGS_DIR / f"{fname}.yaml"
    if not cfg_path.exists():
        # Search subdirectories (e.g. configs/tasks/hetbi/<task>.yaml)
        matches = list(CONFIGS_DIR.rglob(f"{fname}.yaml"))
        if matches:
            cfg_path = matches[0]
        else:
            raise FileNotFoundError(f"No config found at {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, modifying base in place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


class BaseTask(gymnasium.Env):
    """Every task in the benchmark implements exactly this interface."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    # --- subclasses declare these as class vars ---
    TASK_ID: str = ""
    DESCRIPTION: str = ""
    MAX_EPISODE_STEPS: int = 500
    CONFIG_FILE: str | None = None  # override to share one config across many tasks

    def __init__(
        self,
        render_mode: str | None = None,
        seed: int = 0,
        obs_mode: str = "combo",
        config_overrides: dict | None = None,
        max_episode_steps: int | None = None,
        **kwargs,
    ) -> None:
        self.cfg = load_task_config(
            self.TASK_ID, config_file=self.CONFIG_FILE, overrides=config_overrides,
        )
        obs_cfg = self.cfg.get("obs", {})

        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.viewer_camera = self.cfg.get("viewer_camera", None)

        # Free-camera perspective used by the MuJoCo viewer (render_mode="human").
        # Defaults give a natural diagonal view; override per task via YAML
        # viewer_perspective: {azimuth, elevation, distance, lookat}.
        _vp = self.cfg.get("viewer_perspective", {})
        self._viewer_cam_config = {
            "azimuth":   _vp.get("azimuth",   45.0),
            "elevation": _vp.get("elevation", -25.0),
            "distance":  _vp.get("distance",   1.5),
            "lookat":    _vp.get("lookat",    [0.0, 0.0, 0.85]),
        }

        self.reward_type = self.cfg.get("reward_type", "dense")
        self.action_mode = self.cfg.get("action_mode", "cartesian_delta")
        if max_episode_steps is not None:
            self.MAX_EPISODE_STEPS = max_episode_steps
            self.cfg["max_episode_steps"] = max_episode_steps
        else:
            self.MAX_EPISODE_STEPS = self.cfg.get("max_episode_steps", self.MAX_EPISODE_STEPS)
        self.control_freq = self.cfg.get("control_freq", 20)
        self.seed = seed
        self._step_count = 0
        self._viewer_maximized = False

        # ── Controller configuration ────────────────────────────────
        # Parsed from the YAML ``controller`` section and turned into the
        # ``controller_configs`` list that robosuite.make() expects.
        self.controller_configs = self._build_controller_configs()

        # Parse which modalities are enabled from config
        self.proprio_enabled = obs_cfg.get("proprio", {}).get("enabled", False)
        self.proprio_keys = obs_cfg.get("proprio", {}).get("keys", [])
        self.object_state_enabled = obs_cfg.get("object_state", {}).get("enabled", False)
        self.ft_enabled = obs_cfg.get("force_torque", {}).get("enabled", False)
        self.ft_keys = obs_cfg.get("force_torque", {}).get("keys", [])

        tactile_cfg = obs_cfg.get("tactile", {})
        self.tactile_enabled = tactile_cfg.get("enabled", False)
        self.tactile_fingers = tactile_cfg.get("fingers", ["right", "left"])
        self.tactile_rows = tactile_cfg.get("rows", 24)
        self.tactile_cols = tactile_cfg.get("cols", 16)
        self.tactile_spacing = tactile_cfg.get("spacing", 0.01)

        # Parse cameras — collect only the enabled ones
        # Cameras can be either:
        #   1. Built-in: just a robosuite_name reference
        #   2. Custom: defined with pos, quat, parent_body — injected into MuJoCo after env build
        self.camera_cfgs = {}  # label → {robosuite_name, width, height, depth, ...}
        self._custom_cameras = {}  # label → {pos, quat, fovy, parent_body}
        cameras_cfg = obs_cfg.get("cameras", {})
        for label, cam in cameras_cfg.items():
            if cam.get("enabled", False):
                cam_name = cam.get("robosuite_name", label)
                self.camera_cfgs[label] = {
                    "robosuite_name": cam_name,
                    "width": cam.get("width", 84),
                    "height": cam.get("height", 84),
                    "depth": cam.get("depth", False),
                }
                # If pos is specified, it's a custom camera we need to inject
                if "pos" in cam:
                    self._custom_cameras[label] = {
                        "name": cam_name,
                        "pos": cam["pos"],
                        "quat": cam.get("quat", [1, 0, 0, 0]),
                        "fovy": cam.get("fovy", 60),
                        "parent_body": cam.get("parent_body", "world"),
                    }

        self.has_cameras = len(self.camera_cfgs) > 0

        # Camera render throttle: MuJoCo offscreen rendering is the most
        # expensive part of the obs pipeline. Nothing downstream consumes
        # frames faster than ``recording.save_freq.cameras`` (dashboards
        # cap at 25 FPS too), so we only call ``sim.render`` every
        # ``_camera_render_skip`` simulation steps and reuse the cached
        # frame in between. At 100 Hz control × 25 Hz cameras this cuts
        # render calls 4x.
        cam_save_freq = self.cfg.get("recording", {}).get("save_freq", {}).get(
            "cameras", self.control_freq
        )
        cam_save_freq = max(1, int(cam_save_freq))
        self._camera_render_skip = max(1, self.control_freq // cam_save_freq)
        self._camera_cache: dict[str, np.ndarray] = {}
        self._camera_render_tick = 0

        # ── Dataset save config ────────────────────────────────────────
        # Parse which modalities / extras should be written to the dataset.
        # Each obs modality has a ``save`` flag; extras live under
        # ``recording.save``.
        self._save_proprio = obs_cfg.get("proprio", {}).get("save", True)
        self._save_object_state = obs_cfg.get("object_state", {}).get("save", True)
        self._save_ft = obs_cfg.get("force_torque", {}).get("save", True)
        self._save_tactile = obs_cfg.get("tactile", {}).get("save", True)
        # Per-camera save flags
        self._save_cameras = {}
        for label, cam in cameras_cfg.items():
            if cam.get("enabled", False):
                self._save_cameras[label] = cam.get("save", True)

        rec_save = self.cfg.get("recording", {}).get("save", {})
        self._save_extras = {
            "action": rec_save.get("action", True),
            "reward": rec_save.get("reward", True),
            "task_score": rec_save.get("task_score", True),
            "joint_positions": rec_save.get("joint_positions", True),
            "ee_poses": rec_save.get("ee_poses", True),
        }

        self._observation_space = None
        self._action_space = None
        self._env = self._build_env(**kwargs)

        # Record which robots actually received tactile sensors.
        # The dynamic subclass stores this during _load_model().
        self._tactile_robot_indices = getattr(
            self._env, "_tactile_robot_indices", []
        )

        # Inject custom cameras into the MuJoCo model after env is built
        if self._custom_cameras:
            self._inject_custom_cameras()

        # Fix eye_in_hand camera for suction grippers (pump body blocks default placement)
        self._fix_suction_eye_in_hand()
        # Initialise gymnasium.Env internals (np_random, etc.)
        super().__init__()

    # ---- helpers for subclasses to query camera setup ----

    def _camera_names(self) -> list[str]:
        """Robosuite camera name list for enabled cameras."""
        return [c["robosuite_name"] for c in self.camera_cfgs.values()]

    def _camera_heights(self) -> list[int]:
        return [c["height"] for c in self.camera_cfgs.values()]

    def _camera_widths(self) -> list[int]:
        return [c["width"] for c in self.camera_cfgs.values()]

    # ---- controller config builder ----

    def _build_controller_configs(self) -> list[dict] | None:
        ctrl_cfg = self.cfg.get("controller")
        if ctrl_cfg is None:
            return None

        # Normalise to a list (one entry per robot)
        if isinstance(ctrl_cfg, dict):
            ctrl_cfgs = [ctrl_cfg]
        else:
            ctrl_cfgs = list(ctrl_cfg)

        # Load the canonical per-type part configs from robosuite
        _parts_dir = Path(robosuite.__file__).parent / "controllers" / "config" / "default" / "parts"

        result = []
        for cfg in ctrl_cfgs:
            robot_name = cfg.get("robot", "Panda")
            base = load_composite_controller_config(robot=robot_name)
            arm_key = cfg.get("arm_key", next(iter(base["body_parts"])))

            ctrl_type = cfg.get("type", base["body_parts"][arm_key].get("type", "OSC_POSE"))

            # If the requested type differs from the default, replace the arm
            # section with the canonical part config for that type so we don't
            # carry over incompatible keys (e.g. OSC keys when using IK_POSE).
            default_type = base["body_parts"][arm_key].get("type", "OSC_POSE")
            if ctrl_type != default_type:
                type_to_file = {
                    "OSC_POSE": "osc_pose.json",
                    "OSC_POSITION": "osc_position.json",
                    "IK_POSE": "ik_pose.json",
                    "JOINT_POSITION": "joint_position.json",
                    "JOINT_VELOCITY": "joint_velocity.json",
                    "JOINT_TORQUE": "joint_torque.json",
                }
                part_file = _parts_dir / type_to_file.get(ctrl_type, "osc_pose.json")
                with open(part_file) as f:
                    arm = json.load(f)
                # Preserve the gripper config from the default
                gripper_cfg = base["body_parts"][arm_key].get("gripper")
                if gripper_cfg:
                    arm["gripper"] = gripper_cfg
                base["body_parts"][arm_key] = arm
            else:
                arm = base["body_parts"][arm_key]

            # Apply any additional tuning parameters from YAML
            _tunable_keys = (
                "kp", "damping_ratio", "impedance_mode",
                "input_type", "input_ref_frame", "interpolation",
                "ramp_ratio", "kp_limits", "damping_ratio_limits",
                "output_max", "output_min",
                "ik_pos_limit", "ik_ori_limit", "control_delta",
                "uncouple_pos_ori", "position_limits",
                "orientation_limits",
            )
            for key in _tunable_keys:
                if key in cfg:
                    base["body_parts"][arm_key][key] = cfg[key]

            result.append(base)

        return result

    def _inject_custom_cameras(self) -> None:
        """Override camera pos/quat/fovy for cameras defined with custom placement in the YAML.

        The camera must already exist in the MuJoCo model (i.e. robosuite_name must
        reference a built-in camera). This method overrides its position, orientation,
        and field of view after the env is constructed.
        """
        sim = self._env.sim
        for label, cam_def in self._custom_cameras.items():
            cam_name = cam_def["name"]
            try:
                cam_id = sim.model.camera_name2id(cam_name)
            except (ValueError, KeyError):
                raise ValueError(
                    f"Custom camera '{label}': robosuite_name '{cam_name}' "
                    f"not found in model. Available: "
                    f"{[sim.model.camera_id2name(i) for i in range(sim.model.ncam)]}"
                )
            sim.model.cam_pos[cam_id] = cam_def["pos"]
            sim.model.cam_quat[cam_id] = cam_def["quat"]
            sim.model.cam_fovy[cam_id] = cam_def["fovy"]

    def _fix_suction_eye_in_hand(self) -> None:
        """Adjust eye_in_hand camera position for robots with suction grippers.

        The default eye_in_hand camera sits at pos="0.05 0 0" on the
        right_hand body.  The suction pump extends ~11 cm along the -Y axis
        of right_hand (due to the gripper's -90° X rotation), blocking the
        default camera's downward (+Z) view.  This shifts the camera to
        pos="0.1 -0.05 0" which clears the pump body from the field of view.
        """
        gripper_types = self.cfg.get("grippers", [])
        if isinstance(gripper_types, str):
            gripper_types = [gripper_types]

        suction_indices = [
            idx for idx, gtype in enumerate(gripper_types)
            if "Suction" in gtype
        ]
        if not suction_indices:
            return

        import re

        def _patch_suction_cam_xml(xml_string):
            for idx in suction_indices:
                cam_name = f"robot{idx}_eye_in_hand"
                # Match the camera element by name and replace its pos attribute
                pattern = (
                    r'(<camera[^>]*name\s*=\s*"'
                    + re.escape(cam_name)
                    + r'"[^>]*)\bpos\s*=\s*"[^"]*"'
                )
                replacement = r'\1pos="0.1 -0.05 0"'
                xml_string = re.sub(pattern, replacement, xml_string)
            return xml_string

        self._env.set_xml_processor(_patch_suction_cam_xml)
        # Recompile the model so the patched camera position takes effect.
        # The first super().__init__() may have already created a
        # MjRenderContextOffscreen against the old sim. _initialize_sim()
        # below replaces self.sim with a fresh MjSim, but the old GL
        # context can linger and corrupt the new offscreen FBO state
        # (symptom: cameras return banded / black frames). Tear it down
        # explicitly before recompiling, then rebuild via _reset_internal
        # so the new offscreen context is bound to the new sim.
        import gc
        old_sim = getattr(self._env, "sim", None)
        if old_sim is not None and getattr(old_sim, "_render_context_offscreen", None) is not None:
            try:
                del old_sim._render_context_offscreen
            except AttributeError:
                pass
            old_sim._render_context_offscreen = None
        old_viewer = getattr(self._env, "viewer", None)
        if old_viewer is not None and self._env.renderer == "mujoco":
            try:
                self._env._destroy_viewer()
            except AttributeError:
                pass
        gc.collect()
        self._env._initialize_sim()
        self._env._reset_internal()

    # ── Robosuite env factory (auto-wraps with tactile) ─────────────

    def _make_robosuite_env(self, env_name, **kwargs):
        """Create a robosuite env, wrapping with tactile if enabled.

        Subclasses should call this instead of ``suite.make()`` so that
        tactile injection happens automatically during ``_load_model``.

        Centralises two render fixes that previously caused intermittent
        black camera frames:

        1. Pins MuJoCo offscreen rendering to a specific GPU via
           ``render_gpu_device_id``. The id is read from
           ``MUJOCO_EGL_DEVICE_ID`` (or ``IB_RENDER_GPU_DEVICE_ID``)
           and defaults to ``0``. Set either env var to ``-1`` to
           restore the robosuite default (auto/CPU).
        2. Guards against running the GLFW onscreen viewer
           (``has_renderer=True``) and the offscreen renderer in the
           same process — that combination shares a context group
           between two GL paths and is the most common cause of the
           "sometimes the camera returns black" symptom. To run both
           anyway, export ``MUJOCO_GL=egl`` (the offscreen path then
           gets its own EGL context, isolated from GLFW) and set
           ``IB_ALLOW_VIEWER_WITH_OFFSCREEN=1`` to silence the warning.
        """
        env_cls = REGISTERED_ENVS[env_name]

        if self.tactile_enabled:
            env_cls = make_tactile_env_class(
                env_cls,
                tactile_rows=self.tactile_rows,
                tactile_cols=self.tactile_cols,
                tactile_spacing=self.tactile_spacing,
            )

        if "render_gpu_device_id" not in kwargs:
            gpu_id_str = (
                os.environ.get("IB_RENDER_GPU_DEVICE_ID")
                or os.environ.get("MUJOCO_EGL_DEVICE_ID")
                or "0"
            )
            try:
                kwargs["render_gpu_device_id"] = int(gpu_id_str)
            except ValueError:
                kwargs["render_gpu_device_id"] = 0

        if (
            kwargs.get("has_renderer")
            and kwargs.get("has_offscreen_renderer")
            and os.environ.get("MUJOCO_GL", "").lower() != "egl"
            and os.environ.get("IB_ALLOW_VIEWER_WITH_OFFSCREEN") != "1"
        ):
            warnings.warn(
                "Running has_renderer=True together with has_offscreen_renderer=True "
                "on the GLFW backend is a known cause of intermittent black camera "
                "frames. Either drop the viewer (render_mode=None), export "
                "MUJOCO_GL=egl so the offscreen path uses its own EGL context, or "
                "set IB_ALLOW_VIEWER_WITH_OFFSCREEN=1 to silence this warning.",
                RuntimeWarning,
                stacklevel=2,
            )

        return env_cls(**kwargs)

    # ── Tactile helpers ───────────────────────────────────────────────

    def get_tactile_readings(self, finger: str, robot_idx: int = 0) -> np.ndarray:
        """Return ``(rows, cols)`` touch array for *finger* on *robot_idx*.

        Returns a zero array when tactile is disabled or the robot has no
        finger gripper, so callers never need to branch.
        """
        if not self.tactile_enabled or robot_idx not in self._tactile_robot_indices:
            return np.zeros((self.tactile_rows, self.tactile_cols))
        return _tactile_readings(
            self._env.sim, robot_idx, finger,
            self.tactile_rows, self.tactile_cols,
        )

    def get_all_tactile_readings(self) -> dict[int, dict[str, np.ndarray]]:
        """Return ``{robot_idx: {finger: ndarray}}`` for every tactile-equipped robot."""
        result = {}
        for robot_idx in self._tactile_robot_indices:
            result[robot_idx] = {
                f: self.get_tactile_readings(f, robot_idx)
                for f in self.tactile_fingers
            }
        return result

    def _inject_tactile_obs(self, raw_obs: dict[str, np.ndarray]) -> None:
        """Write ``(rows, cols)`` tactile matrices into *raw_obs* so that
        ``_build_obs_from_config`` picks them up automatically.

        Keys written: ``robot{idx}_tactile_{finger}`` for each
        tactile-equipped robot and enabled finger.  Values are
        ``(rows, cols)`` numpy arrays.
        """
        if not self.tactile_enabled:
            return
        for robot_idx in self._tactile_robot_indices:
            for finger in self.tactile_fingers:
                key = f"robot{robot_idx}_tactile_{finger}"
                if key not in raw_obs:
                    raw_obs[key] = self.get_tactile_readings(finger, robot_idx)

    # ---- 4 methods every subclass must implement ----

    @abstractmethod
    def _build_env(self, **kwargs) -> object:
        """Return the raw robosuite env."""

    @abstractmethod
    def _get_obs(self, raw_obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Convert raw robosuite obs → standardised obs dict."""

    @abstractmethod
    def _compute_task_score(self, raw_obs: dict[str, np.ndarray]) -> float:
        """Return normalised 0–1 task completion score."""

    @abstractmethod
    def _check_termination(self, raw_obs: dict[str, np.ndarray]) -> bool:
        """Return True when episode should end (success or failure)."""

    # ---- shared obs builder (subclasses call this from _get_obs) ----

    def _build_obs_from_config(self, raw_obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Build the obs dict based on what the YAML config enables."""
        obs = {}

        # Proprio — store each key separately so downstream code can
        # access e.g. obs["robot0_eef_pos"] directly.
        if self.proprio_enabled:
            for key in self.proprio_keys:
                if key in raw_obs:
                    obs[key] = np.atleast_1d(raw_obs[key])

        # Cameras — render directly via sim.render() instead of using
        # raw_obs images.  Robosuite's observable sampling has a float-
        # precision bug that causes it to skip camera re-renders after
        # many steps, returning stale/frozen images.  Direct rendering
        # bypasses this entirely and also handles the OpenGL vertical
        # flip (origin at bottom-left → top-left).
        #
        # Throttling: render at ``recording.save_freq.cameras`` Hz rather
        # than the full control frequency — this is the slowest consumer
        # and anything faster is wasted work. Intermediate steps reuse
        # frames from ``self._camera_cache``. Always render when the
        # cache is empty (first call after reset).
        if self.obs_mode in ("rgb", "combo"):
            sim = getattr(self._env, "sim", None)
            render_now = (
                not self._camera_cache
                or (self._camera_render_tick % self._camera_render_skip == 0)
            )
            for label, cam_cfg in self.camera_cfgs.items():
                rs_name = cam_cfg["robosuite_name"]
                w, h = cam_cfg["width"], cam_cfg["height"]
                rgb_key = f"rgb_{label}"
                depth_key_out = f"depth_{label}"
                if render_now:
                    if sim is not None:
                        img = sim.render(width=w, height=h, camera_name=rs_name, depth=False)
                        self._camera_cache[rgb_key] = img[::-1]
                    else:
                        img_key = f"{rs_name}_image"
                        if img_key in raw_obs:
                            self._camera_cache[rgb_key] = raw_obs[img_key][::-1]
                    if cam_cfg["depth"]:
                        if sim is not None:
                            _, depth = sim.render(width=w, height=h, camera_name=rs_name, depth=True)
                            self._camera_cache[depth_key_out] = depth[::-1]
                        else:
                            depth_key = f"{rs_name}_depth"
                            if depth_key in raw_obs:
                                self._camera_cache[depth_key_out] = raw_obs[depth_key][::-1]
                if rgb_key in self._camera_cache:
                    obs[rgb_key] = self._camera_cache[rgb_key]
                if cam_cfg["depth"] and depth_key_out in self._camera_cache:
                    obs[depth_key_out] = self._camera_cache[depth_key_out]
            self._camera_render_tick += 1

        # Object state
        if self.object_state_enabled and "object-state" in raw_obs:
            obs["object_state"] = raw_obs["object-state"]

        # Force/torque
        if self.ft_enabled:
            parts = []
            world_parts = []
            for key in self.ft_keys:
                if key in raw_obs:
                    parts.append(raw_obs[key])
                wkey = f"{key}_world_force"
                if wkey in raw_obs:
                    world_parts.append(raw_obs[wkey])
            if parts:
                obs["force_torque"] = np.concatenate(parts)
            if world_parts:
                # Shape (N_robots * 3,) — per-robot world-frame [Fx, Fy, Fz].
                obs["force_torque_world"] = np.concatenate(world_parts)
        
        if self.tactile_enabled:
            for robot_idx in self._tactile_robot_indices:
                for finger in self.tactile_fingers:
                    key = f"robot{robot_idx}_tactile_{finger}"
                    if key in raw_obs:
                        obs[key] = raw_obs[key]

        return obs

    # ---- dataset save filter ----

    def filter_obs_for_saving(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return a copy of *obs* containing only keys whose ``save``
        flag is ``true`` in the YAML config."""
        out = {}
        for key, val in obs.items():
            if key.startswith("rgb_") or key.startswith("depth_"):
                # Camera key — check per-camera save flag
                # key is e.g. "rgb_exo_front_cam" → label "exo_front_cam"
                label = key.split("_", 1)[1] if "_" in key else key
                if self._save_cameras.get(label, True):
                    out[key] = val
            elif key == "force_torque":
                if self._save_ft:
                    out[key] = val
            elif key in ("object_state", "object_pos", "object_quat",
                         "cube_pos", "cube_quat") or key.startswith("obj_"):
                if self._save_object_state:
                    out[key] = val
            elif "_tactile_" in key:
                if self._save_tactile:
                    out[key] = val
            elif key in ("joint_positions", "ee_poses"):
                if self._save_extras.get(key, True):
                    out[key] = val
            else:
                # Proprio keys (robot0_eef_pos, etc.)
                if self._save_proprio:
                    out[key] = val
        return out

    @property
    def save_extras(self) -> dict[str, bool]:
        """Return the ``recording.save`` config for non-obs fields."""
        return self._save_extras

    # ---- F/T sensor extraction ----

    def _inject_ft_sensor(self, raw_obs: dict[str, np.ndarray]) -> None:
        """Extract force/torque from the robot and inject into raw_obs.

        Robosuite's ``ee_force`` / ``ee_torque`` are read from a MuJoCo
        force sensor at the eef site, so the wrench is expressed in the
        **eef local frame**. For weight-sensing UIs we also rotate the
        force into the **world frame** so gravity always loads the same
        axis — the z-component of the world-frame force is then a direct
        proxy for held object weight regardless of gripper orientation.

        Stores a 6D vector (Fx, Fy, Fz, Tx, Ty, Tz) in the eef frame
        under the configured sensor key, plus a world-frame 3D force
        under ``robot{i}_ft_sensor_world_force`` for the dashboard.
        """
        if not self.ft_enabled:
            return
        for i, robot in enumerate(self._env.robots):
            key = f"robot{i}_ft_sensor"
            try:
                ee_force = robot.ee_force
                ee_torque = robot.ee_torque
                # ee_force/ee_torque are dicts keyed by arm name for multi-arm
                # robots, or plain arrays for single-arm — normalise to array
                if isinstance(ee_force, dict):
                    arm_name = next(iter(ee_force.keys()))
                    ee_force = ee_force[arm_name]
                    ee_torque = ee_torque[arm_name] if isinstance(ee_torque, dict) else ee_torque
                else:
                    arm_name = None
                force = np.array(ee_force, dtype=np.float64).flatten()
                torque = np.array(ee_torque, dtype=np.float64).flatten()
                raw_obs[key] = np.concatenate([force, torque])

                # World-frame force — rotate sensor-frame force by the
                # sensor site's world orientation. The force sensor sits
                # at site ``gripper*_ft_frame`` (dereferenced via the
                # gripper's ``important_sensors['force_ee']`` name).
                try:
                    sim = self._env.sim
                    grip = robot.gripper[arm_name] if arm_name is not None else robot.gripper
                    sens_name = grip.important_sensors["force_ee"]
                    sid = sim.model.sensor_name2id(sens_name)
                    site_id = sim.model.sensor_objid[sid]
                    site_name = sim.model.site_id2name(site_id)
                    R = sim.data.get_site_xmat(site_name)  # world<-sensor
                    raw_obs[f"{key}_world_force"] = R @ force
                except (KeyError, ValueError) as e:
                    logger.debug("World-frame F/T rotation skipped for robot%d: %s", i, e)
            except (AttributeError, RuntimeError) as e:
                logger.warning("Failed to read F/T sensor for robot%d: %s", i, e)

    # ---- base class handles everything else ----

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, np.ndarray], dict]:
        # Let gymnasium.Env.reset() initialise np_random when a seed is given
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
        raw_obs = self._env.reset()
        self._step_count = 0
        # Force a fresh camera render on the first obs of the new episode.
        self._camera_cache.clear()
        self._camera_render_tick = 0
        # Re-inject custom cameras — robosuite's reset() rebuilds the MuJoCo
        # model, which overwrites any camera pos/quat/fovy we set earlier.
        if self._custom_cameras:
            self._inject_custom_cameras()
        self._inject_ft_sensor(raw_obs)
        self._inject_tactile_obs(raw_obs)
        obs = self._get_obs(raw_obs)
        info = {"task_score": 0.0, "step": 0}
        return obs, info

    def _get_sim_observations(self, force_update: bool = False) -> dict[str, np.ndarray]:
        """Retrieve observations from the underlying robosuite environment.

        Wraps robosuite's internal ``_get_observations`` so all callers
        in this codebase go through a single, documented point of contact
        with the private robosuite API.  The ``force_update`` flag is
        required after scene mutations (object teleport, visual randomisation)
        to bypass robosuite's per-timestep observable cache.

        Parameters
        ----------
        force_update:
            When ``True``, bypasses robosuite's observable cache and
            forces a full re-evaluation of all sensors.
        """
        return self._env._get_observations(force_update=force_update)

    def _check_grasp(
        self,
        gripper,
        object_geoms: list[str] | str,
    ) -> bool:
        """Check whether *gripper* is currently grasping *object_geoms*.

        Wraps robosuite's internal ``_check_grasp`` so task subclasses
        do not reference robosuite private API directly.

        Parameters
        ----------
        gripper:
            Robosuite gripper model (e.g. ``self._env.robots[0].gripper``).
        object_geoms:
            A single geom name or list of geom names belonging to the
            target object.

        Returns
        -------
        bool
            ``True`` if the gripper is in firm contact with the object.
        """
        return self._env._check_grasp(gripper=gripper, object_geoms=object_geoms)

    def _rebuild_obs(self) -> dict[str, np.ndarray]:
        """Re-render and rebuild the obs dict from current sim state.

        Subclass ``reset()`` methods call ``super().reset()`` first (which
        renders an obs), then mutate the scene — reposition objects,
        change masses, place static markers, etc. The obs returned by
        the super-call is therefore stale: the first saved demo frame
        shows the pre-mutation scene. Calling this helper at the end of
        a subclass reset returns a fresh obs that reflects all changes.

        Cameras honour the render-throttle cache; this call forces a
        fresh frame by clearing the cache first.
        """
        self._camera_cache.clear()
        raw_obs = self._get_sim_observations(force_update=True)
        self._inject_ft_sensor(raw_obs)
        self._inject_tactile_obs(raw_obs)
        return self._get_obs(raw_obs)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        raw_obs, reward, done, info = self._env.step(action)
        self._step_count += 1
        self._inject_ft_sensor(raw_obs)
        self._inject_tactile_obs(raw_obs)
        obs = self._get_obs(raw_obs)
        task_score = self._compute_task_score(raw_obs)
        terminated = bool(self._check_termination(raw_obs))
        truncated = bool(self._step_count >= self.MAX_EPISODE_STEPS)
        info.update({"task_score": task_score, "step": self._step_count})
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            self._env.render()
            if not self._viewer_maximized:
                self._try_maximize_mujoco_viewer()
                self._viewer_maximized = True
            return None
        if self.render_mode == "rgb_array":
            sim = getattr(self._env, "sim", None)
            if sim is None:
                return None
            # Use the first configured camera, fall back to agentview
            if self.camera_cfgs:
                first = next(iter(self.camera_cfgs.values()))
                cam_name = first["robosuite_name"]
                w, h = first["width"], first["height"]
            else:
                cam_name, w, h = "agentview", 256, 256
            frame = sim.render(width=w, height=h, camera_name=cam_name, depth=False)
            return frame[::-1]  # flip vertical (MuJoCo origin is bottom-left)
        return None

    def _try_maximize_mujoco_viewer(self) -> None:
        import subprocess
        import time
        time.sleep(0.2)  # let the window manager register the new window
        target = getattr(self, "_viewer_target_geometry", None)
        try:
            if target:
                x, y, w, h = target
                subprocess.run(
                    ["wmctrl", "-r", "MuJoCo", "-e", f"0,{x},{y},{w},{h}"],
                    check=False, capture_output=True, timeout=2,
                )
            else:
                subprocess.run(
                    ["wmctrl", "-r", "MuJoCo", "-b", "add,maximized_vert,maximized_horz"],
                    check=False, capture_output=True, timeout=2,
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def task_score(self) -> float:
        """Return the current task score in [0, 1] without stepping."""
        raw_obs = self._get_sim_observations()
        return self._compute_task_score(raw_obs)

    def close(self) -> None:
        self._env.close()

    @property
    def observation_space(self) -> spaces.Dict:
        """Gymnasium observation space, built lazily on first access.

        Each key maps to a :class:`~gymnasium.spaces.Box` with the shape
        of the corresponding observation array.  Camera observations use
        ``dtype=uint8`` with bounds ``[0, 255]``; all others use
        ``dtype=float64`` with ``[-inf, inf]`` bounds.
        """
        if self._observation_space is None:
            obs, _ = self.reset()
            box_spaces = {}
            for k, v in obs.items():
                if v.dtype == np.uint8:
                    box_spaces[k] = spaces.Box(low=0, high=255, shape=v.shape, dtype=np.uint8)
                else:
                    box_spaces[k] = spaces.Box(
                        low=-np.inf, high=np.inf, shape=v.shape, dtype=np.float64
                    )
            self._observation_space = spaces.Dict(box_spaces)
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: spaces.Space) -> None:
        """Override the cached observation space (used in tests)."""
        self._observation_space = space

    @property
    def action_space(self) -> spaces.Box:
        """Gymnasium action space derived from the robosuite action spec.

        Returns a :class:`~gymnasium.spaces.Box` whose ``low`` and ``high``
        match the underlying robosuite environment's action bounds.
        """
        if self._action_space is None:
            low, high = self._env.action_spec
            self._action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self._action_space

    @action_space.setter
    def action_space(self, space: spaces.Space) -> None:
        """Override the cached action space (used in tests)."""
        self._action_space = space
