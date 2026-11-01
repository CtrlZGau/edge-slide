"""IMBench — physics-based robotics benchmark & dataset library.

Quick start::

    import imbench as ib

    task = ib.make("ib-p1-edge-slide-v1", obs_mode="combo", seed=0)
    obs, info = task.reset()
    obs, reward, terminated, truncated, info = task.step(task.action_space.sample())
    task.close()

See :func:`~imbench.registry.list_tasks` for all available task IDs.
"""

__version__ = "0.1.0"

from imbench.registry import list_tasks, make  # noqa: F401
from imbench.policy.base_policy import PolicyInterface  # noqa: F401


__all__ = ["make", "list_tasks", "PolicyInterface", "__version__"]
