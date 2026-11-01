"""Mirror Pick Place — pick up a cube off the table to a target lift height.

Task ID: ``ib-m-mirror-pick-place-v1``
"""

import numpy as np
from imbench.tasks.prehensile.single_arm_base import SingleArmBase


class MirrorPickPlaceV1(SingleArmBase):
    TASK_ID = "ib-m-mirror-pick-place-v1"
    DESCRIPTION = (
        "Single arm: pick up a cube lying on the table and lift it to a "
        "target height. Cube position and color are randomized each reset."
    )

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._cube_size = task_cfg.get("cube_size", 0.02)
        self._cube_mass = task_cfg.get("cube_mass", 0.05)
        self._lift_height = task_cfg.get("lift_height", 0.1)

    def _make_objects(self):
        return [
            {"type": "box", "name": "cube",
             "size": [self._cube_size] * 3,
             "rgba": [0.2, 0.5, 0.8, 1.0],
             "density": self._cube_mass / (8 * self._cube_size ** 3),
             "friction": [1.0, 0.005, 0.0001],
             "joints": "default"},
        ]

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Randomize cube color (independent of the visual-randomization
        # flag so this task always has a varying cube color).
        sim = self._env.sim
        cube_body_id = sim.model.body_name2id("cube_main")
        geom_start = sim.model.body_geomadr[cube_body_id]
        geom_count = sim.model.body_geomnum[cube_body_id]
        rgba = self._random_object_color()
        for gi in range(geom_start, geom_start + geom_count):
            sim.model.geom_rgba[gi] = rgba

        sim.forward()
        obs = self._rebuild_obs()
        self._flip_cameras(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._flip_cameras(obs)
        return obs, reward, terminated, truncated, info

    def _flip_cameras(self, obs):
        # Flip every camera frame both horizontally and vertically
        # (equivalent to a 180° rotation in the image plane). The
        # bird-eye camera is exempt — its top-down view is unaffected
        # by the in-plane flip the task uses to mirror the scene.
        for key in list(obs.keys()):
            if not (key.startswith("rgb_") or key.startswith("depth_")):
                continue
            if "bird" in key:
                continue
            obs[key] = obs[key][::-1, ::-1]

    def _compute_task_score(self, raw_obs):
        if "cube_pos" not in raw_obs:
            return 0.0
        cube_z = raw_obs["cube_pos"][2]
        lift = cube_z - self._table_height
        return 1.0 if lift >= self._lift_height else 0.0

    def _check_termination(self, raw_obs):
        if self._compute_task_score(raw_obs) >= self._success_score_threshold:
            return True
        return self._object_fell_off_table(raw_obs)
