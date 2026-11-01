"""Task registry and factory for IMBench.

Provides :data:`TASK_REGISTRY` (a mapping of task IDs to task classes),
:func:`make` to instantiate a task by ID, and :func:`list_tasks` to
enumerate all registered task IDs.

Task IDs follow the naming scheme ``ib-<category>-<name>-v<version>``
where category encodes the physics phenomenon tested:

- ``ib-p1-*`` — P1: Geometry-constrained grasping
- ``ib-p2-*`` — P2: Dynamics and trajectory prediction
- ``ib-p3-*`` — P3: Causal chains (indirect action)
- ``ib-p4-*`` — P4: Tool use (augmented reach)
- ``ib-p5-*`` — P5: Hidden state (active discovery)
- ``ib-p6-*`` — P6: Reactive replanning under perturbation
- ``ib-p7-*`` — P7: Stability and equilibrium construction
- ``ib-m-*``  — M: Miscellaneous / utility tasks
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imbench.tasks.base_task import BaseTask

from imbench.tasks.prehensile.edge_slide_v1 import EdgeSlideV1
from imbench.tasks.prehensile.cube_pick_v1 import CubePickV1
from imbench.tasks.prehensile.mirror_pick_place_v1 import MirrorPickPlaceV1

# Trimmed for the edge-slide submission: only the tasks used by this repo.
TASK_REGISTRY = {
    "ib-p1-edge-slide-v1": EdgeSlideV1,   # main task (all evals and training)
    "ib-p1-cube-pick-v1": CubePickV1,     # cube pick (third task of the multi-task DP)
    "ib-m-mirror-pick-place-v1": MirrorPickPlaceV1,  # VLA set-up sanity check (cube pick, report Part 2)
}


def make(task_id: str, **kwargs) -> "BaseTask":
    """Instantiate an IMBench task by ID.

    Parameters
    ----------
    task_id:
        A registered task identifier, e.g. ``"ib-p1-edge-slide-v1"``.
        Call :func:`list_tasks` to see all available IDs.
    **kwargs:
        Forwarded to the task constructor.  Common options:

        - ``obs_mode`` — ``"combo"`` (proprio + cameras) or ``"proprio"``
        - ``render_mode`` — ``None``, ``"human"``, or ``"rgb_array"``
        - ``seed`` — integer seed for reproducibility
        - ``max_episode_steps`` — episode length cap

    Returns
    -------
    BaseTask
        The fully constructed, ready-to-use task instance.

    Raises
    ------
    ValueError
        If *task_id* is not in :data:`TASK_REGISTRY`.
    """
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task_id}'. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[task_id](**kwargs)


def list_tasks() -> list[str]:
    """Return a sorted list of all registered task IDs."""
    return list(TASK_REGISTRY.keys())


def _register_gymnasium_envs() -> None:
    """Register every task with the gymnasium registry so ``gymnasium.make()`` works.

    Each task's class (or partial) is used directly as the ``entry_point``.
    When ``gymnasium.make("ib-p1-edge-slide-v1", render_mode="rgb_array")``
    is called, gymnasium invokes ``entry_point(**kwargs)`` — i.e. the task
    class constructor — with whatever kwargs were passed to ``make()``.
    """
    import gymnasium

    for task_id, cls in TASK_REGISTRY.items():
        if task_id in gymnasium.envs.registry:
            continue
        gymnasium.register(
            id=task_id,
            entry_point=cls,  # gymnasium calls cls(**kwargs) to create the env
        )


# Register on import so ``gymnasium.make("ib-p1-...")`` works out of the box.
_register_gymnasium_envs()
