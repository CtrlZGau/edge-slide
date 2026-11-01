"""Cube Pick v1: top-down grasp and lift of a box on the edge-slide table.

Same table, robot, cameras, scoring and termination as ``ib-p1-edge-slide-v1``
(success = box lifted ``lift_height`` above the table with gripper contact;
failure if it falls off). Only the object differs: a box of random length,
width and **height**, thick enough for a top-down parallel-jaw grasp across
its short horizontal side.

The box height is randomised like length and width: the model is compiled at
the upper bound of the range and the geom is shrunk in ``reset()``.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from imbench.tasks.prehensile.edge_slide_v1 import EdgeSlideV1


class CubePickV1(EdgeSlideV1):
    TASK_ID = "ib-p1-cube-pick-v1"
    DESCRIPTION = "Single arm: Pick up the cube from the table"

    def _parse_task_config(self):
        super()._parse_task_config()
        rand_cfg = self.cfg.get("randomization", {})
        ht_rand = rand_cfg.get("object_half_height", {}).get("range")
        self._book_height_range = tuple(ht_rand) if ht_rand else None
        if self._book_height_range:
            # _build_env compiles the geom with _book_half_height: use the upper bound.
            self._book_half_height = self._book_height_range[1]

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if not self._book_height_range:
            return obs, info

        # Continue the task RNG stream (already seeded by EdgeSlideV1.reset).
        half_ht = float(self._rng.uniform(*self._book_height_range))
        sim = self._env.sim
        bid = self._find_body_id(sim, "cube")
        for i in range(sim.model.ngeom):
            if sim.model.geom_bodyid[i] == bid:
                sim.model.geom_size[i][2] = half_ht
        x, y = sim.data.body_xpos[bid][:2]
        yaw = Rotation.from_quat(np.roll(sim.data.body_xquat[bid], -1)).as_euler("xyz")[2]
        self._teleport_cube(sim, float(x), float(y), self._table_height + half_ht + 0.001, float(yaw))
        sim.forward()

        # Re-observe; force_update=True so cameras/observables aren't stale.
        self._camera_cache.clear()
        raw_obs = self._get_sim_observations(force_update=True)
        raw_obs["cube_pos"] = np.array(sim.data.body_xpos[bid])
        self._inject_ft_sensor(raw_obs)
        self._inject_tactile_obs(raw_obs)
        return self._get_obs(raw_obs), info
