"""pi0.5 policy client — queries an openpi policy server over websocket.

The model runs in a separate process (openpi's ``scripts/serve_policy.py``)
because openpi pins its own JAX / Python stack. This class only adapts
IMBench observations to the LIBERO input format and replays action chunks.

    # terminal 1 (openpi repo)
    uv run scripts/serve_policy.py --env LIBERO

    # terminal 2 (IMBench)
    policy = Pi05Policy(prompt="pick up the flat plate from the table")
"""

from __future__ import annotations

import collections
import math

import numpy as np

from imbench.policy.base_policy import PolicyInterface


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite (x, y, z, w) quaternion → axis-angle, as in openpi's LIBERO client."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class Pi05Policy(PolicyInterface):
    """Adapter from IMBench ``combo`` observations to a pi0.5 LIBERO server.

    Parameters
    ----------
    prompt:
        Language instruction sent with every query.
    host, port:
        Address of the openpi policy server.
    main_cam, wrist_cam:
        IMBench camera labels mapped to ``observation/image`` and
        ``observation/wrist_image``.
    mirror:
        LIBERO training frames are raw robosuite renders rotated 180°.
        IMBench already flips rows, so only a horizontal mirror remains.
    replan_steps:
        Number of actions executed from each predicted chunk before
        querying the server again.
    state_pos_offset:
        Added to ``robot0_eef_pos`` so the state is expressed in LIBERO's
        world frame. IMBench's Panda base sits at x=-0.41, LIBERO's at
        x=-0.66, hence the default (-0.25, 0, 0).
    """

    def __init__(
        self,
        prompt: str,
        host: str = "localhost",
        port: int = 8000,
        main_cam: str = "libero_cam",
        wrist_cam: str = "gripper_cam",
        mirror: bool = True,
        resize: int = 224,
        replan_steps: int = 5,
        state_pos_offset: tuple[float, float, float] = (-0.25, 0.0, 0.0),
    ):
        from openpi_client import websocket_client_policy

        self.prompt = prompt
        self.main_cam = main_cam
        self.wrist_cam = wrist_cam
        self.mirror = mirror
        self.resize = resize
        self.replan_steps = replan_steps
        self.state_pos_offset = np.asarray(state_pos_offset, dtype=np.float64)
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self._plan: collections.deque = collections.deque()
        # Last model inputs, exposed for rollout videos.
        self.last_main_img: np.ndarray | None = None
        self.last_wrist_img: np.ndarray | None = None

    def _prep(self, img: np.ndarray) -> np.ndarray:
        from openpi_client import image_tools

        if self.mirror:
            img = img[:, ::-1]
        img = np.ascontiguousarray(img)
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, self.resize, self.resize)
        )

    def reset(self) -> None:
        self._plan.clear()

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        self.last_main_img = self._prep(obs[f"rgb_{self.main_cam}"])
        self.last_wrist_img = self._prep(obs[f"rgb_{self.wrist_cam}"])
        if not self._plan:
            state = np.concatenate([
                obs["robot0_eef_pos"] + self.state_pos_offset,
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ])
            chunk = self._client.infer({
                "observation/image": self.last_main_img,
                "observation/wrist_image": self.last_wrist_img,
                "observation/state": state,
                "prompt": self.prompt,
            })["actions"]
            self._plan.extend(chunk[: self.replan_steps])
        return np.clip(np.asarray(self._plan.popleft(), dtype=np.float64), -1.0, 1.0)
