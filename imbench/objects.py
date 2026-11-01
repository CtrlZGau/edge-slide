"""Robosuite-compatible wrappers for YCB dataset objects and extra fixtures.

``YCBObject``  – loads any object from ``deps/ycb/ycb/<name>/`` using the
    CoACD collision STLs + textured OBJ visual from the elpis-lab YCB repo.

``ExtrasObject`` – loads primitive-geometry fixtures (shelf, rack, box, tray)
    from ``deps/extras/<name>.xml``.

Both inherit from robosuite's ``MujocoXMLObject`` and are drop-in replacements
wherever robosuite expects a ``MujocoObject``.
"""

from __future__ import annotations

import math
import os
import struct
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
from robosuite.models.objects.objects import MujocoXMLObject

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
YCB_DATASET_DIR = _PROJECT_ROOT / "deps" / "ycb" / "ycb"
EXTRAS_DIR = _PROJECT_ROOT / "deps" / "extras"


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# Initial bounds for STL bounding-box accumulation; large enough to be
# dominated by any real geometry, but still finite for numpy comparisons.
_BOUND_INIT_MIN: float = 1e9
_BOUND_INIT_MAX: float = -1e9

# Default YCB object material density used for mass estimation when no
# explicit density is provided (kg/m³, roughly wood/plastic composite).
_DEFAULT_YCB_DENSITY: float = 200.0


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _stl_bounds(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) from a binary STL file."""
    with open(filepath, "rb") as f:
        f.read(80)  # header
        (num_tris,) = struct.unpack("<I", f.read(4))
        verts = np.empty((num_tris * 3, 3), dtype=np.float32)
        for i in range(num_tris):
            f.read(12)  # normal
            for j in range(3):
                verts[i * 3 + j] = struct.unpack("<fff", f.read(12))
            f.read(2)  # attribute
    return verts.min(axis=0).astype(np.float64), verts.max(axis=0).astype(np.float64)


def _compute_sites(mesh_dir: str) -> dict:
    """Compute bottom_site, top_site, horizontal_radius_site from collision STLs."""
    stls = sorted(
        f for f in os.listdir(mesh_dir) if f.endswith(".stl")
    )
    if not stls:
        # fallback
        return {"bottom": "0 0 -0.05", "top": "0 0 0.05", "hrad": "0.05 0.05 0"}

    all_min = np.array([_BOUND_INIT_MIN] * 3)
    all_max = np.array([_BOUND_INIT_MAX] * 3)
    for s in stls:
        mn, mx = _stl_bounds(os.path.join(mesh_dir, s))
        all_min = np.minimum(all_min, mn)
        all_max = np.maximum(all_max, mx)

    sz = all_max - all_min
    hrad = max(sz[0], sz[1]) / 2.0
    return {
        "bottom": f"0 0 {all_min[2]:.5f}",
        "top": f"0 0 {all_max[2]:.5f}",
        "hrad": f"{hrad:.5f} {hrad:.5f} 0",
    }


def _generate_robosuite_xml(
    ycb_name: str,
    mesh_dir: str,
    sites: dict,
    density: float = _DEFAULT_YCB_DENSITY,
) -> str:
    """Generate a robosuite-compatible MuJoCo XML string for a YCB object.

    Robosuite expectations:
      - ``worldbody/body/body[@name='object']``
      - collision geoms in group 0
      - visual geoms in group 1
      - sites: ``bottom_site``, ``top_site``, ``horizontal_radius_site``
      - no freejoint (``MujocoXMLObject`` adds one)
    """
    # Discover mesh files
    visual_obj = os.path.join(mesh_dir, "textured.obj")
    collision_stls = sorted(
        f for f in os.listdir(mesh_dir) if f.endswith(".stl")
    )
    texture_png = os.path.join(mesh_dir, "texture_map.png")

    has_texture = os.path.isfile(texture_png)

    lines = [f'<mujoco model="{ycb_name}">']
    lines.append("  <asset>")

    # Visual mesh — generic names; robosuite's naming_prefix controls the final name
    lines.append(
        f'    <mesh name="vis_mesh" file="{visual_obj}"/>'
    )
    # Collision meshes
    for i, stl in enumerate(collision_stls):
        lines.append(
            f'    <mesh name="col_mesh_{i}" file="{os.path.join(mesh_dir, stl)}"/>'
        )
    # Texture + material
    if has_texture:
        lines.append(
            f'    <texture name="tex" type="2d" file="{texture_png}"/>'
        )
        lines.append(
            '    <material name="mat" texture="tex" texrepeat="1 1" texuniform="false"/>'
        )
    lines.append("  </asset>")

    lines.append("  <worldbody>")
    lines.append("    <body>")
    lines.append('      <body name="object">')

    # Visual geom (group 1)
    mat_attr = 'material="mat"' if has_texture else 'rgba="0.6 0.6 0.6 1"'
    lines.append(
        f'        <geom name="vis" type="mesh" mesh="vis_mesh" '
        f'{mat_attr} group="1" contype="0" conaffinity="0"/>'
    )

    # Collision geoms (group 0). rgba alpha=0 so they don't overlay the
    # textured visual mesh; COACD hulls don't match the visual surface
    # exactly and would otherwise render as gray bulges through it.
    for i in range(len(collision_stls)):
        lines.append(
            f'        <geom name="col_{i}" type="mesh" mesh="col_mesh_{i}" '
            f'group="0" density="{density}" '
            f'solimp="0.998 0.998 0.001" solref="0.001 1" '
            f'friction="0.95 0.3 0.1" condim="4" '
            f'rgba="0.5 0.5 0.5 0"/>'
        )

    lines.append("      </body>")

    # Sites (outside "object" body, inside outer body — robosuite convention)
    lines.append(
        f'      <site rgba="0 0 0 0" size="0.005" pos="{sites["bottom"]}" name="bottom_site"/>'
    )
    lines.append(
        f'      <site rgba="0 0 0 0" size="0.005" pos="{sites["top"]}" name="top_site"/>'
    )
    lines.append(
        f'      <site rgba="0 0 0 0" size="0.005" pos="{sites["hrad"]}" name="horizontal_radius_site"/>'
    )

    lines.append("    </body>")
    lines.append("  </worldbody>")
    lines.append("</mujoco>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  YCBObject
# ---------------------------------------------------------------------------

class YCBObject(MujocoXMLObject):
    """A robosuite ``MujocoXMLObject`` backed by an elpis-lab YCB mesh.

    Parameters
    ----------
    name : str
        Instance name used by robosuite's naming/prefixing system.
    ycb_name : str
        Directory name under ``deps/ycb/ycb/`` (e.g. ``"mug"``, ``"bowl"``).
    density : float
        MuJoCo density for collision geoms.
    joints : str | list | None
        Joint specification (default: free joint).
    """

    def __init__(
        self,
        name: str,
        ycb_name: str,
        density: float = 200.0,
        joints="default",
        obj_type: str = "all",
        duplicate_collision_geoms: bool = True,
    ):
        mesh_dir = str(YCB_DATASET_DIR / ycb_name)
        if not os.path.isdir(mesh_dir):
            raise FileNotFoundError(
                f"YCB object '{ycb_name}' not found at {mesh_dir}. "
                f"Available: {sorted(d for d in os.listdir(YCB_DATASET_DIR) if os.path.isdir(YCB_DATASET_DIR / d))}"
            )

        sites = _compute_sites(mesh_dir)
        xml_str = _generate_robosuite_xml(ycb_name, mesh_dir, sites, density=density)

        # Write to a temp file so MujocoXMLObject can parse it
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", prefix=f"ycb_{ycb_name}_", delete=False
        )
        tmp.write(xml_str)
        tmp.flush()
        tmp.close()

        try:
            super().__init__(
                fname=tmp.name,
                name=name,
                joints=joints,
                obj_type=obj_type,
                duplicate_collision_geoms=duplicate_collision_geoms,
            )
        finally:
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
#  ExtrasObject
# ---------------------------------------------------------------------------

class ExtrasObject(MujocoXMLObject):
    """A robosuite ``MujocoXMLObject`` loaded from ``deps/extras/<name>.xml``.

    These are primitive-geometry fixtures (shelves, racks, boxes, trays) used
    as static targets in het-bi tasks.

    Parameters
    ----------
    name : str
        Instance name for robosuite prefixing.
    extras_name : str
        Filename stem under ``deps/extras/`` (e.g. ``"shelf"``, ``"rack"``).
    joints : str | list | None
        ``None`` for a static fixture (no joint), ``"default"`` for free body.
    """

    def __init__(
        self,
        name: str,
        extras_name: str,
        joints=None,
        obj_type: str = "all",
        duplicate_collision_geoms: bool = True,
    ):
        xml_path = str(EXTRAS_DIR / f"{extras_name}.xml")
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(
                f"Extras object '{extras_name}' not found at {xml_path}. "
                f"Available: {sorted(p.stem for p in EXTRAS_DIR.glob('*.xml'))}"
            )
        super().__init__(
            fname=xml_path,
            name=name,
            joints=joints,
            obj_type=obj_type,
            duplicate_collision_geoms=duplicate_collision_geoms,
        )


# ---------------------------------------------------------------------------
#  PendulumObject
# ---------------------------------------------------------------------------

def _generate_pendulum_xml(
    stand_height: float = 0.25,
    arm_length: float = 0.20,
    bob_size: float = 0.02,
    bob_rgba: tuple = (0.2, 0.2, 0.8, 1.0),
    stand_rgba: tuple = (0.5, 0.5, 0.5, 1.0),
    damping: float = 0.001,
    bob_shape: str = "sphere",
) -> str:
    """Generate MuJoCo XML for a pendulum with a static stand and hinge arm.

    Layout: vertical stand post → horizontal bracket along +X → hinge at
    bracket tip.  Bob hangs down and swings in the YZ plane (horizontal
    from the robot's viewpoint).  The stand and bracket are far from the
    swing arc so the bob never collides with them.

    ``bob_shape`` is ``"sphere"`` (``bob_size`` = radius) or ``"box"``
    (``bob_size`` = half-edge).
    """
    sr = " ".join(f"{c}" for c in stand_rgba)
    br = " ".join(f"{c}" for c in bob_rgba)

    # Bracket extends along -X (toward robot) so when the stand is placed
    # at a far-x position, the bob hangs closer to the robot and the stand
    # is at the far end — completely out of the bob's swing trajectory.
    bracket_len = 0.10

    if bob_shape == "sphere":
        bob_size_attr = f"{bob_size}"
    elif bob_shape == "box":
        bob_size_attr = f"{bob_size} {bob_size} {bob_size}"
    else:
        raise ValueError(f"bob_shape must be 'sphere' or 'box', got {bob_shape!r}")

    return f"""<mujoco model="pendulum">
  <worldbody>
    <body>
      <body name="object">
        <!-- Vertical stand post — at the origin (far end from robot) -->
        <geom name="stand_post" type="cylinder"
              fromto="0 0 0 0 0 {stand_height}"
              size="0.01" rgba="{sr}" group="1" mass="2.0"
              contype="0" conaffinity="0"/>
        <!-- Horizontal bracket extending along -X (toward robot) from top of stand -->
        <geom name="bracket" type="cylinder"
              fromto="0 0 {stand_height} {-bracket_len} 0 {stand_height}"
              size="0.008" rgba="{sr}" group="1" mass="0.5"
              contype="0" conaffinity="0"/>
        <!-- Pendulum arm: hinge at end of bracket (closer to robot), axis
             along X for left-right swing in YZ plane -->
        <body name="arm" pos="{-bracket_len} 0 {stand_height}">
          <joint name="hinge" type="hinge" axis="1 0 0"
                 damping="{damping}" frictionloss="0"/>
          <!-- Pivot marker (visual only) -->
          <geom name="pivot" type="sphere" size="0.01"
                rgba="0.4 0.4 0.4 1" group="1" mass="0.01"
                contype="0" conaffinity="0"/>
          <!-- Rod hanging downward -->
          <geom name="rod" type="cylinder"
                fromto="0 0 0 0 0 {-arm_length}"
                size="0.004" rgba="0.6 0.6 0.6 1" group="1" mass="0.02"
                contype="0" conaffinity="0"/>
          <!-- Bob (sphere or box at the end of the rod — has collision for grasping) -->
          <geom name="bob" type="{bob_shape}"
                pos="0 0 {-arm_length - bob_size}"
                size="{bob_size_attr}"
                rgba="{br}" group="0" mass="0.1"/>
        </body>
        <!-- Sites for robosuite -->
        <site name="bottom_site" pos="0 0 -0.001" size="0.001" rgba="0 0 0 0"/>
        <site name="top_site" pos="0 0 {stand_height + 0.005}" size="0.001" rgba="0 0 0 0"/>
        <site name="horizontal_radius_site" pos="0.02 0 0" size="0.001" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


class PendulumObject(MujocoXMLObject):
    """A static stand with a freely-swinging pendulum arm via hinge joint.

    The root body (stand) is welded to world (joints=None).
    The child body (arm + bob) swings via a MuJoCo hinge joint.

    Parameters
    ----------
    name : str
        Instance name for robosuite prefixing.
    stand_height : float
        Height of the vertical stand above its base position.
    arm_length : float
        Length of the pendulum rod.
    bob_size : float
        Sphere radius (or box half-edge) of the bob.
    bob_rgba : tuple
        RGBA colour for the bob.
    damping : float
        Hinge joint damping (0 = ideal/frictionless pendulum).
    bob_shape : str
        ``"sphere"`` or ``"box"``.
    """

    def __init__(
        self,
        name: str,
        stand_height: float = 0.25,
        arm_length: float = 0.20,
        bob_size: float = 0.02,
        bob_rgba: tuple = (0.2, 0.2, 0.8, 1.0),
        damping: float = 0.001,
        bob_shape: str = "sphere",
    ):
        xml_str = _generate_pendulum_xml(
            stand_height=stand_height,
            arm_length=arm_length,
            bob_size=bob_size,
            bob_rgba=bob_rgba,
            damping=damping,
            bob_shape=bob_shape,
        )

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", prefix="pendulum_", delete=False
        )
        tmp.write(xml_str)
        tmp.flush()
        tmp.close()

        try:
            super().__init__(
                fname=tmp.name,
                name=name,
                joints=None,  # stand is welded; child hinge is preserved
                obj_type="all",
                duplicate_collision_geoms=True,
            )
        finally:
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
#  TiltingInclineObject  (kinematically driven hinged plate)
# ---------------------------------------------------------------------------

def _generate_tilting_incline_xml(
    plate_length: float = 0.55,
    plate_half_width: float = 0.225,
    plate_half_thickness: float = 0.004,
    plate_rgba: tuple = (0.45, 0.55, 0.65, 1.0),
    plate_friction: tuple = (1.0, 0.005, 0.0001),
    damping: float = 100.0,
    frictionloss: float = 0.0,
) -> str:
    """Generate MuJoCo XML for a hinged incline plate.

    Layout: a static welded "anchor" body holds a child "plate" body
    on a hinge joint about local -Y. The plate's low edge sits at the
    body origin and extends along +x (up-slope direction). Positive
    joint angle lifts the high (+x) end. The task drives the joint
    kinematically by writing qpos/qvel each step.
    """
    pr = " ".join(f"{c}" for c in plate_rgba)
    pf = " ".join(f"{c}" for c in plate_friction)

    return f"""<mujoco model="tilting_incline">
  <worldbody>
    <body>
      <body name="object">
        <body name="plate" pos="0 0 0">
          <joint name="incline_hinge" type="hinge" axis="0 -1 0"
                 damping="{damping}" frictionloss="{frictionloss}"
                 limited="true" range="-90 90"/>
          <geom name="plate" type="box"
                size="{plate_length / 2} {plate_half_width} {plate_half_thickness}"
                pos="{plate_length / 2} 0 {plate_half_thickness}"
                rgba="{pr}" group="0" mass="50.0"
                friction="{pf}"
                contype="1" conaffinity="1"/>
        </body>
        <site name="bottom_site" pos="0 0 -0.001" size="0.001" rgba="0 0 0 0"/>
        <site name="top_site" pos="0 0 0.05" size="0.001" rgba="0 0 0 0"/>
        <site name="horizontal_radius_site" pos="{plate_length} 0 0" size="0.001" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


class TiltingInclineObject(MujocoXMLObject):
    """A welded anchor + hinged plate, driven by a position-servo actuator.

    The plate's low edge sits at the body origin (the hinge axis) and
    extends along body +x to ``plate_length``. The hinge axis is body
    -Y so positive joint angle lifts the high end.

    The task drives the joint by writing ``sim.data.ctrl[servo_id] =
    target_rad`` each step. The actuator's high kp keeps tracking
    rigid under contact loads.
    """

    def __init__(
        self,
        name: str,
        plate_length: float = 0.55,
        plate_half_width: float = 0.225,
        plate_half_thickness: float = 0.004,
        plate_rgba: tuple = (0.45, 0.55, 0.65, 1.0),
        plate_friction: tuple = (1.0, 0.005, 0.0001),
        damping: float = 100.0,
        frictionloss: float = 0.0,
    ):
        xml_str = _generate_tilting_incline_xml(
            plate_length=plate_length,
            plate_half_width=plate_half_width,
            plate_half_thickness=plate_half_thickness,
            plate_rgba=plate_rgba,
            plate_friction=plate_friction,
            damping=damping,
            frictionloss=frictionloss,
        )

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", prefix="tilting_incline_", delete=False
        )
        tmp.write(xml_str)
        tmp.flush()
        tmp.close()

        try:
            super().__init__(
                fname=tmp.name,
                name=name,
                joints=None,  # anchor is welded; plate hinge is preserved
                obj_type="all",
                duplicate_collision_geoms=True,
            )
        finally:
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
#  SeesawObject
# ---------------------------------------------------------------------------

def _generate_seesaw_xml(
    plank_half_length: float = 0.225,
    plank_half_width: float = 0.03,
    plank_half_thickness: float = 0.004,
    fulcrum_height: float = 0.04,
    stopper_height: float = 0.015,
    plank_rgba: tuple = (0.3, 0.7, 0.3, 1.0),
    fulcrum_rgba: tuple = (0.5, 0.5, 0.5, 1.0),
    plank_friction: tuple = (0.15, 0.005, 0.0001),
    damping: float = 0.01,
    frictionloss: float = 0.001,
) -> str:
    """Generate MuJoCo XML for a seesaw with a hinge joint.

    Layout: static fulcrum wedge on the table, with a plank body attached
    via a hinge joint (axis along Y) at the top of the fulcrum.  The plank
    rotates freely around the fulcrum.  Stoppers at both ends prevent objects
    from sliding off.
    """
    pr = " ".join(f"{c}" for c in plank_rgba)
    fr = " ".join(f"{c}" for c in fulcrum_rgba)
    pf = " ".join(f"{c}" for c in plank_friction)

    return f"""<mujoco model="seesaw">
  <worldbody>
    <body>
      <body name="object">
        <!-- Static fulcrum wedge (visual only — hinge handles constraint) -->
        <geom name="fulcrum" type="box"
              size="0.015 {plank_half_width} {fulcrum_height / 2}"
              pos="0 0 {fulcrum_height / 2}"
              rgba="{fr}" group="1" mass="5.0"
              contype="0" conaffinity="0"/>
        <!-- Plank body with hinge at fulcrum top -->
        <body name="plank" pos="0 0 {fulcrum_height}">
          <joint name="seesaw_hinge" type="hinge" axis="0 1 0"
                 damping="{damping}" frictionloss="{frictionloss}"
                 limited="true" range="-30 30"/>
          <!-- Main plank beam -->
          <geom name="plank_beam" type="box"
                size="{plank_half_length} {plank_half_width} {plank_half_thickness}"
                pos="0 0 {plank_half_thickness}"
                rgba="{pr}" group="0" mass="0.3"
                friction="{pf}"/>
          <!-- Stopper at near end (-x) -->
          <geom name="stopper_near" type="box"
                size="0.005 {plank_half_width} {stopper_height}"
                pos="{-plank_half_length + 0.005} 0 {plank_half_thickness * 2 + stopper_height}"
                rgba="{fr}" group="0" mass="0.05"
                friction="1.0 0.005 0.0001"/>
          <!-- Stopper at far end (+x) -->
          <geom name="stopper_far" type="box"
                size="0.005 {plank_half_width} {stopper_height}"
                pos="{plank_half_length - 0.005} 0 {plank_half_thickness * 2 + stopper_height}"
                rgba="{fr}" group="0" mass="0.05"
                friction="1.0 0.005 0.0001"/>
        </body>
        <!-- Sites for robosuite -->
        <site name="bottom_site" pos="0 0 -0.001" size="0.001" rgba="0 0 0 0"/>
        <site name="top_site" pos="0 0 {fulcrum_height + plank_half_thickness * 2 + stopper_height * 2 + 0.005}" size="0.001" rgba="0 0 0 0"/>
        <site name="horizontal_radius_site" pos="{plank_half_length} 0 0" size="0.001" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def _generate_prism_xml(
    base_half_width: float = 0.025,
    half_depth: float = 0.02,
    half_height: float = 0.03,
    rgba: tuple = (0.3, 0.8, 0.3, 1.0),
    density: float = 500,
    friction: tuple = (0.8, 0.005, 0.0001),
) -> str:
    """Generate MuJoCo XML for a triangular prism using a convex mesh.

    The prism cross-section is an isosceles triangle in XZ, extruded
    along Y.  Vertices are centred at the body origin so placement
    works the same as a ``BoxObject`` of similar half-extents.
    """
    r = " ".join(f"{c}" for c in rgba)
    fr = " ".join(f"{c}" for c in friction)
    bw, hd, hh = base_half_width, half_depth, half_height

    # 6 vertices: 2 triangular faces (front/back in Y)
    verts = (
        f"{-bw} {-hd} {-hh}  {bw} {-hd} {-hh}  {0} {-hd} {hh}  "
        f"{-bw} {hd} {-hh}   {bw} {hd} {-hh}    {0} {hd} {hh}"
    )

    return f"""<mujoco model="prism">
  <asset>
    <mesh name="prism_mesh" vertex="{verts}"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <!-- Visual geom (group 1) -->
        <geom name="g0_vis" type="mesh" mesh="prism_mesh"
              rgba="{r}" group="1"
              contype="0" conaffinity="0" mass="0.0001"/>
        <!-- Collision geom (group 0). Softer contact params + condim=4 so
             polygonal-face contacts don't chatter when stacked, and the
             prism can rotate on edge without tangential jitter. -->
        <geom name="g0" type="mesh" mesh="prism_mesh"
              rgba="{r}" density="{density}"
              friction="{fr}" group="0"
              contype="1" conaffinity="1"
              condim="4"
              solref="0.01 1.5" solimp="0.9 0.95 0.001"/>
      </body>
      <site name="bottom_site" pos="0 0 {-hh}" size="0.001" rgba="0 0 0 0"/>
      <site name="top_site" pos="0 0 {hh}" size="0.001" rgba="0 0 0 0"/>
      <site name="horizontal_radius_site" pos="{bw} 0 0" size="0.001" rgba="0 0 0 0"/>
    </body>
  </worldbody>
</mujoco>"""


class TriangularPrismObject(MujocoXMLObject):
    """A triangular prism built from an inline convex mesh.

    Parameters
    ----------
    name : str
        Unique object name (robosuite naming).
    size : list
        ``[base_half_width, half_depth, half_height]``
    rgba, density, friction
        Appearance / physics properties.
    """

    def __init__(
        self,
        name: str,
        size=(0.025, 0.02, 0.03),
        rgba=(0.3, 0.8, 0.3, 1.0),
        density: float = 500,
        friction=(0.8, 0.005, 0.0001),
        joints=None,
    ):
        xml_str = _generate_prism_xml(
            base_half_width=size[0],
            half_depth=size[1],
            half_height=size[2],
            rgba=rgba,
            density=density,
            friction=friction,
        )

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", prefix="prism_", delete=False
        )
        tmp.write(xml_str)
        tmp.flush()
        tmp.close()

        joint_arg = joints if joints is not None else [{"type": "free"}]

        try:
            super().__init__(
                fname=tmp.name,
                name=name,
                joints=joint_arg,
                obj_type="all",
                duplicate_collision_geoms=False,
            )
        finally:
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
#  TorusObject  (ring from capsule segments)
# ---------------------------------------------------------------------------

def _generate_torus_xml(
    major_radius: float = 0.035,
    minor_radius: float = 0.005,
    n_segments: int = 16,
    rgba: tuple = (0.8, 0.6, 0.1, 1.0),
    density: float = 600,
    friction: tuple = (0.5, 0.005, 0.0001),
) -> str:
    """Generate MuJoCo XML for a torus (ring) using capsule segments.

    ``n_segments`` capsules are arranged in a circle of ``major_radius``.
    Each capsule has tube radius ``minor_radius``.
    """
    r = " ".join(f"{c}" for c in rgba)
    fr = " ".join(f"{c}" for c in friction)

    total_vol = 2 * math.pi ** 2 * major_radius * minor_radius ** 2
    total_mass = total_vol * density
    mass_per_seg = total_mass / n_segments

    geom_lines = []
    for i in range(n_segments):
        t0 = 2 * math.pi * i / n_segments
        t1 = 2 * math.pi * (i + 1) / n_segments
        x0 = major_radius * math.cos(t0)
        y0 = major_radius * math.sin(t0)
        x1 = major_radius * math.cos(t1)
        y1 = major_radius * math.sin(t1)
        geom_lines.append(
            f'        <geom name="seg_{i}_vis" type="capsule" '
            f'fromto="{x0:.6f} {y0:.6f} 0 {x1:.6f} {y1:.6f} 0" '
            f'size="{minor_radius}" rgba="{r}" group="1" '
            f'contype="0" conaffinity="0" mass="0.0001"/>'
        )
        geom_lines.append(
            f'        <geom name="seg_{i}" type="capsule" '
            f'fromto="{x0:.6f} {y0:.6f} 0 {x1:.6f} {y1:.6f} 0" '
            f'size="{minor_radius}" rgba="{r}" mass="{mass_per_seg:.6f}" '
            f'friction="{fr}" group="0" contype="1" conaffinity="1"/>'
        )

    geoms = "\n".join(geom_lines)
    outer = major_radius + minor_radius
    return f"""<mujoco model="torus">
  <worldbody>
    <body>
      <body name="object">
{geoms}
      </body>
      <site name="bottom_site" pos="0 0 {-minor_radius}" size="0.001" rgba="0 0 0 0"/>
      <site name="top_site" pos="0 0 {minor_radius}" size="0.001" rgba="0 0 0 0"/>
      <site name="horizontal_radius_site" pos="{outer} 0 0" size="0.001" rgba="0 0 0 0"/>
    </body>
  </worldbody>
</mujoco>"""


class TorusObject(MujocoXMLObject):
    """A torus (ring) built from capsule segments arranged in a circle."""

    def __init__(
        self,
        name: str,
        major_radius: float = 0.035,
        minor_radius: float = 0.005,
        n_segments: int = 16,
        rgba=(0.8, 0.6, 0.1, 1.0),
        density: float = 600,
        friction=(0.5, 0.005, 0.0001),
        joints=None,
    ):
        xml_str = _generate_torus_xml(
            major_radius=major_radius,
            minor_radius=minor_radius,
            n_segments=n_segments,
            rgba=rgba,
            density=density,
            friction=friction,
        )

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", prefix="torus_", delete=False
        )
        tmp.write(xml_str)
        tmp.flush()
        tmp.close()

        joint_arg = joints if joints is not None else [{"type": "free"}]

        try:
            super().__init__(
                fname=tmp.name,
                name=name,
                joints=joint_arg,
                obj_type="all",
                duplicate_collision_geoms=False,
            )
        finally:
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
#  SeesawObject  (fulcrum + hinge-jointed plank)
# ---------------------------------------------------------------------------

class SeesawObject(MujocoXMLObject):
    """A fulcrum with a plank body attached via hinge joint (seesaw).

    The root body (fulcrum) is welded to world (joints=None).
    The child body (plank + stoppers) pivots via a hinge joint around Y.
    """

    def __init__(
        self,
        name: str,
        plank_half_length: float = 0.225,
        plank_half_width: float = 0.03,
        plank_half_thickness: float = 0.004,
        fulcrum_height: float = 0.04,
        stopper_height: float = 0.015,
        plank_rgba: tuple = (0.3, 0.7, 0.3, 1.0),
        plank_friction: tuple = (0.15, 0.005, 0.0001),
        damping: float = 0.01,
        frictionloss: float = 0.001,
    ):
        xml_str = _generate_seesaw_xml(
            plank_half_length=plank_half_length,
            plank_half_width=plank_half_width,
            plank_half_thickness=plank_half_thickness,
            fulcrum_height=fulcrum_height,
            stopper_height=stopper_height,
            plank_rgba=plank_rgba,
            plank_friction=plank_friction,
            damping=damping,
            frictionloss=frictionloss,
        )

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", prefix="seesaw_", delete=False
        )
        tmp.write(xml_str)
        tmp.flush()
        tmp.close()

        try:
            super().__init__(
                fname=tmp.name,
                name=name,
                joints=None,  # fulcrum is welded; child hinge is preserved
                obj_type="all",
                duplicate_collision_geoms=True,
            )
        finally:
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
#  BalancePlatformObject  (narrow ridge for balancing tasks)
# ---------------------------------------------------------------------------

def _generate_balance_platform_xml(
    ridge_half_length: float = 0.06,
    ridge_half_width: float = 0.003,
    ridge_height: float = 0.02,
    base_half_size: tuple = (0.04, 0.04, 0.005),
    ridge_rgba: tuple = (0.4, 0.4, 0.4, 1.0),
    base_rgba: tuple = (0.55, 0.55, 0.55, 1.0),
    friction: tuple = (1.0, 0.005, 0.0001),
) -> str:
    """Generate MuJoCo XML for a narrow balance platform.

    Layout: a flat base plate on the table with a thin ridge on top.
    The rod must be balanced vertically on the narrow ridge edge.
    """
    rr = " ".join(f"{c}" for c in ridge_rgba)
    br = " ".join(f"{c}" for c in base_rgba)
    fr = " ".join(f"{c}" for c in friction)
    bx, by, bz = base_half_size

    return f"""<mujoco model="balance_platform">
  <worldbody>
    <body>
      <body name="object">
        <!-- Base plate -->
        <geom name="base_plate" type="box"
              size="{bx} {by} {bz}"
              pos="0 0 {bz}"
              rgba="{br}" group="0" mass="2.0"
              friction="{fr}"/>
        <!-- Narrow ridge on top -->
        <geom name="ridge" type="box"
              size="{ridge_half_length} {ridge_half_width} {ridge_height / 2}"
              pos="0 0 {bz * 2 + ridge_height / 2}"
              rgba="{rr}" group="0" mass="0.5"
              friction="{fr}"/>
        <!-- Sites for robosuite -->
        <site name="bottom_site" pos="0 0 -0.001" size="0.001" rgba="0 0 0 0"/>
        <site name="top_site" pos="0 0 {bz * 2 + ridge_height + 0.005}" size="0.001" rgba="0 0 0 0"/>
        <site name="horizontal_radius_site" pos="{max(bx, ridge_half_length)} 0 0" size="0.001" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


class BalancePlatformObject(MujocoXMLObject):
    """A narrow ridge on a flat base — the target surface for balancing a rod.

    The platform is static (welded to world, ``joints=None``).
    """

    def __init__(
        self,
        name: str,
        ridge_half_length: float = 0.06,
        ridge_half_width: float = 0.003,
        ridge_height: float = 0.02,
        base_half_size: tuple = (0.04, 0.04, 0.005),
        ridge_rgba: tuple = (0.4, 0.4, 0.4, 1.0),
        base_rgba: tuple = (0.55, 0.55, 0.55, 1.0),
        friction: tuple = (1.0, 0.005, 0.0001),
    ):
        xml_str = _generate_balance_platform_xml(
            ridge_half_length=ridge_half_length,
            ridge_half_width=ridge_half_width,
            ridge_height=ridge_height,
            base_half_size=base_half_size,
            ridge_rgba=ridge_rgba,
            base_rgba=base_rgba,
            friction=friction,
        )

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", prefix="balance_platform_", delete=False
        )
        tmp.write(xml_str)
        tmp.flush()
        tmp.close()

        try:
            super().__init__(
                fname=tmp.name,
                name=name,
                joints=None,
                obj_type="all",
                duplicate_collision_geoms=True,
            )
        finally:
            os.unlink(tmp.name)


