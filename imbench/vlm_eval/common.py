"""Shared helpers: prompt loading and building the multimodal input content."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from imbench.vlm_eval.client import image_part, text_part

PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text()


def observation_content(bundle: dict, bundle_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """Return (API content parts, JSON-safe record of the same parts with image paths/hashes)."""
    bundle_dir = Path(bundle_dir)
    md = bundle["metadata"]
    header = (f"TASK GOAL: {md['task_description']}\n\n"
              f"METADATA (JSON; cite fields as meta:<dotted.path>):\n{json.dumps(md, indent=2)}\n\n"
              f"CAMERA FRAMES follow, {len(bundle['frames'])} images. Each is preceded by its label; "
              f"cite them as cam:<camera_name>:f<frame_idx>.")
    parts, record = [text_part(header)], [text_part(header)]
    for f in bundle["frames"]:
        label = (f"[cam:{f['camera']}:f{f['frame_idx']}] camera={f['camera']} frame_idx={f['frame_idx']} "
                 f"t={f['t_s']:.2f}s ({md['cameras'][f['camera']]})")
        path = bundle_dir / f["file"]
        parts += [text_part(label), image_part(path)]
        record += [text_part(label), {"type": "input_image", "file": f["file"],
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "detail": "high"}]
    return parts, record
