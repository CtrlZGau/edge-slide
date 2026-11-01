"""GR00T N1.7 and MolmoAct2 policy clients (LIBERO checkpoints), pi0.5-compatible.

These mirror :class:`imbench.policy.pi05_policy.Pi05Policy` exactly — same
cameras (``libero_cam`` + ``gripper_cam``), same horizontal mirror, same 8-D
state incl. the LIBERO-frame position offset, same prompt, same chunk replanning — so all three VLAs are evaluated with one protocol: each model's
public LIBERO-finetuned checkpoint driving IMBench at LIBERO's 20 Hz with the
LIBERO action convention (7-D OSC delta in [-1, 1], gripper -1 open / +1 close).

The models run in their own processes/venvs and are reached over ZMQ with
msgpack-numpy (the GR00T server protocol; ``scripts/vla_eval/molmoact2_server.py``
speaks the same protocol).

    # GR00T (Isaac-GR00T repo)
    uv run python gr00t/eval/run_gr00t_server.py \
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
        --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper --port 5555
    # MolmoAct2 (its own venv)
    python scripts/vla_eval/molmoact2_server.py --port 5556

Upstream references (read 2026-10-31):
  * Isaac-GR00T ``gr00t/eval/sim/LIBERO/libero_env.py`` — obs keys, 180° image
    rotation, axis-angle state, gripper post-processing.
  * molmoact2 ``lerobot/processor/env_processor.py`` — 180° rotation, 8-D state;
    ``envs/libero.py`` — actions applied as-is.
"""

from __future__ import annotations

import collections
import math
import time

import numpy as np

from imbench.policy.base_policy import PolicyInterface

# Main camera for all LIBERO-checkpoint evals and the MolmoAct2 fine-tuning data: LIBERO's
# agentview pose shifted +0.25 m in x (LIBERO base x=-0.66 vs IMBench -0.41), on robosuite
# "frontview". Frozen copy of the constant the zero-shot runs used (their run_config.json
# records it); scripts/run_pi05_eval.py no longer exports it.
LIBERO_CAM = {
    "enabled": True, "robosuite_name": "frontview",
    "pos": [0.6586 + 0.25, 0.0, 1.6103], "quat": [0.6380, 0.3048, 0.3048, 0.6380],
    "fovy": 45, "width": 256, "height": 256, "depth": False, "save": False,
}


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite (x, y, z, w) quaternion -> axis-angle (same as LIBERO / openpi / GR00T)."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def resize_with_pad(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Aspect-preserving bilinear resize + zero pad (same as openpi_client.image_tools)."""
    from PIL import Image

    h, w = img.shape[:2]
    ratio = max(w / width, h / height)
    nh, nw = int(h / ratio), int(w / ratio)
    resized = np.asarray(Image.fromarray(img).resize((nw, nh), resample=Image.BILINEAR))
    out = np.zeros((height, width, 3), np.uint8)
    top, left = (height - nh) // 2, (width - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


# ── transport ────────────────────────────────────────────────────────────────

class ZmqMsgpackClient:
    """Minimal REQ client for the GR00T ``PolicyServer`` wire protocol."""

    def __init__(self, host: str, port: int, timeout_ms: int = 60_000):
        import zmq

        self._zmq = zmq
        self.addr = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self.ctx = zmq.Context()
        self._connect()

    def _connect(self):
        zmq = self._zmq
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def call(self, endpoint: str, data=None, requires_input: bool = True):
        import msgpack
        import msgpack_numpy as mnp

        req = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        try:
            self.sock.send(msgpack.packb(req, default=mnp.encode))
            msg = self.sock.recv()
        except self._zmq.error.Again:
            self.sock.close()
            self._connect()  # REQ socket is wedged after a timeout
            raise TimeoutError(f"policy server at {self.addr} timed out on {endpoint!r}")
        if msg == b"ERROR":
            raise RuntimeError(f"policy server at {self.addr} returned ERROR")
        resp = msgpack.unpackb(msg, object_hook=mnp.decode, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"policy server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self.call("ping", requires_input=False)
            return True
        except Exception:
            return False


# ── shared adapter ───────────────────────────────────────────────────────────

class _LiberoChunkPolicy(PolicyInterface):
    """Common obs prep + chunk replay; subclasses implement :meth:`_infer`."""

    name = "vla"

    def __init__(
        self,
        prompt: str,
        host: str,
        port: int,
        main_cam: str = "libero_cam",
        wrist_cam: str = "gripper_cam",
        mirror: bool = True,
        resize: int = 256,
        replan_steps: int = 5,
        state_pos_offset: tuple[float, float, float] = (-0.25, 0.0, 0.0),
    ):
        self.prompt = prompt
        self.main_cam = main_cam
        self.wrist_cam = wrist_cam
        self.mirror = mirror
        self.resize = resize
        self.replan_steps = replan_steps
        # IMBench Panda base x=-0.41, LIBERO's x=-0.66 -> express eef pos in LIBERO's
        # world frame (same as Pi05Policy.state_pos_offset).
        self.state_pos_offset = np.asarray(state_pos_offset, dtype=np.float64)
        self.client = ZmqMsgpackClient(host, port)
        self._plan: collections.deque = collections.deque()
        self.last_main_img: np.ndarray | None = None
        self.last_wrist_img: np.ndarray | None = None
        self.infer_ms: list[float] = []

    def _prep(self, img: np.ndarray) -> np.ndarray:
        # LIBERO frames = raw robosuite render rotated 180°. IMBench already
        # flips rows (top-left origin), so only the horizontal mirror remains.
        if self.mirror:
            img = img[:, ::-1]
        return resize_with_pad(np.ascontiguousarray(img), self.resize, self.resize)

    def _state(self, obs) -> np.ndarray:
        return np.concatenate([
            obs["robot0_eef_pos"] + self.state_pos_offset,
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]).astype(np.float32)

    def _infer(self, main, wrist, state) -> np.ndarray:  # -> (H, 7)
        raise NotImplementedError

    def reset(self, episode_seed: int | None = None) -> None:
        self._plan.clear()
        self.infer_ms = []

    def act(self, obs):
        main = self._prep(obs[f"rgb_{self.main_cam}"])
        wrist = self._prep(obs[f"rgb_{self.wrist_cam}"])
        # What the model sees, at 224 px for run_pi05_eval.compose_frame's side panel.
        self.last_main_img = resize_with_pad(main, 224, 224)
        self.last_wrist_img = resize_with_pad(wrist, 224, 224)
        if not self._plan:
            t0 = time.perf_counter()
            chunk = np.asarray(self._infer(main, wrist, self._state(obs)))
            self.infer_ms.append((time.perf_counter() - t0) * 1e3)
            if chunk.ndim != 2 or chunk.shape[1] != 7:
                raise ValueError(f"{self.name}: expected (H, 7) action chunk, got {chunk.shape}")
            self._plan.extend(chunk[: self.replan_steps])
        return np.clip(np.asarray(self._plan.popleft(), dtype=np.float64), -1.0, 1.0)


# ── GR00T N1.7 ───────────────────────────────────────────────────────────────

_GROOT_AXES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


class GrootLiberoPolicy(_LiberoChunkPolicy):
    """GR00T N1.7 LIBERO checkpoint served with ``--use-sim-policy-wrapper``.

    Gripper post-processing replicates Isaac-GR00T's LIBERO env exactly:
    model gripper in [0, 1] → [-1, 1] → sign → invert.
    """

    name = "groot"

    def __init__(self, prompt, host="localhost", port=5555, **kw):
        kw.setdefault("resize", 256)  # LIBERO renders 256x256
        super().__init__(prompt, host, port, **kw)

    def _infer(self, main, wrist, state):
        b = lambda a: np.asarray(a, dtype=np.float32).reshape(1, 1, -1)  # (B, T, D)
        obs = {
            "video.image": main[None, None],          # (B, T, H, W, C) uint8
            "video.wrist_image": wrist[None, None],
            "state.x": b(state[0]), "state.y": b(state[1]), "state.z": b(state[2]),
            "state.roll": b(state[3]), "state.pitch": b(state[4]), "state.yaw": b(state[5]),
            "state.gripper": b(state[6:8]),
            "annotation.human.action.task_description": [self.prompt],
        }
        action, _info = self.client.call("get_action", {"observation": obs, "options": None})
        chunk = np.concatenate(
            [np.asarray(action[f"action.{k}"], dtype=np.float64)[0] for k in _GROOT_AXES], axis=-1
        )  # (16, 7)
        g = 2.0 * chunk[:, -1] - 1.0   # normalize_gripper_action (orig range [0, 1])
        chunk[:, -1] = -np.sign(g)     # binarize + invert_gripper_action
        return chunk

    def reset(self, episode_seed=None):
        super().reset(episode_seed)
        try:
            self.client.call("reset", {"options": None})
        except Exception:
            pass  # sim wrapper reset is stateless


# ── MolmoAct2 ────────────────────────────────────────────────────────────────

class MolmoAct2LiberoPolicy(_LiberoChunkPolicy):
    """``allenai/MolmoAct2-LIBERO`` via ``scripts/vla_eval/molmoact2_server.py``.

    Actions come back un-normalized (``norm_tag="libero"``) and are applied
    as-is, like MolmoAct2's own LIBERO env.
    """

    name = "molmoact2"

    def __init__(self, prompt, host="localhost", port=5556, **kw):
        kw.setdefault("resize", 256)
        super().__init__(prompt, host, port, **kw)

    def _infer(self, main, wrist, state):
        resp = self.client.call("get_action", {
            "images": [main, wrist], "state": state, "task": self.prompt,
        })
        return np.asarray(resp["actions"], dtype=np.float64)

    def reset(self, episode_seed=None):
        super().reset(episode_seed)
        # Seed the server's sampler so (env seed, repeat) rollouts are reproducible.
        if episode_seed is not None:
            self.client.call("reset", {"seed": int(episode_seed)})
