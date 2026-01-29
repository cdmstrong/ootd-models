"""Runpod serverless handler for OOTD inference service.

This module wraps the local inference & background-removal logic into a
Runpod-compatible handler function.

Input format (queue job JSON):

Infer task (default task_type is "infer"):
{
  "input": {
    "task_type": "infer",                # optional, defaults to "infer"
    "prompt": "....",
    "image_paths": ["person.png", "top.png"],
    "height": 1024,                      # optional
    "width": 1024,                       # optional
    "guidance_scale": 1.0,              # optional
    "num_inference_steps": 10,          # optional
    "remove_background": [false, true]  # optional, per-image flags
  }
}

Notes on remove_background flags:
- If omitted            -> no background removal for any image.
- If a single bool      -> applies to all images (legacy/shortcut).
- If a list[bool]       -> per-image flags; extra images default to False.

Background-removal-only task:
{
  "input": {
    "task_type": "remove_background",
    "image_path": "person.png"
  }
}

Both task types return:
{
  "success": true,
  "image_base64": "<PNG as base64>",
  "error_message": null
}
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List

import base64
import os

import runpod

from bg_removal.remover import remove_background
from infer import run_inference


def _bool_flags_for_images(
    image_paths: List[str],
    remove_background_param: Any,
) -> List[bool]:
    """Normalize remove_background param into per-image bool flags."""
    n = len(image_paths)

    # No parameter -> all False
    if remove_background_param is None:
        return [False] * n

    # Single bool -> apply to all images
    if isinstance(remove_background_param, bool):
        return [remove_background_param] * n

    # List-like -> per-image flags
    if isinstance(remove_background_param, list):
        flags: List[bool] = []
        for idx in range(n):
            if idx < len(remove_background_param):
                flags.append(bool(remove_background_param[idx]))
            else:
                flags.append(False)
        return flags

    # Fallback: treat as "no removal"
    return [False] * n


def _encode_image_file_to_base64(path: str) -> str:
    """Read an image file and return PNG base64 string."""
    from PIL import Image  # local import to avoid unnecessary dependency at import time

    with Image.open(path) as img:
        img = img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _handle_infer(job_input: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """Handle the 'infer' task_type."""
    prompt = job_input.get("prompt")
    image_paths = job_input.get("image_paths")

    if not prompt or not image_paths:
        return {
            "success": False,
            "image_base64": None,
            "error_message": "Both 'prompt' and 'image_paths' are required for infer task.",
        }

    height = int(job_input.get("height", 1024))
    width = int(job_input.get("width", 1024))
    guidance_scale = float(job_input.get("guidance_scale", 1.0))
    num_inference_steps = int(job_input.get("num_inference_steps", 10))

    # Per-image background removal flags
    remove_background_param = job_input.get("remove_background")
    flags = _bool_flags_for_images(image_paths, remove_background_param)

    # Prepare processed image paths (some may have background removed)
    processed_paths: List[str] = []
    tmp_dir = os.path.join("/tmp", "bg_removed", job_id)
    os.makedirs(tmp_dir, exist_ok=True)

    for idx, path in enumerate(image_paths):
        if flags[idx]:
            # Remove background and use the processed image path
            output_path = os.path.join(tmp_dir, f"img_{idx}.png")
            processed_path = remove_background(path, output_path=output_path)
            processed_paths.append(processed_path)
        else:
            processed_paths.append(path)

    try:
        # run_inference already returns base64-encoded PNG string
        image_base64 = run_inference(
            prompt=prompt,
            image_paths=processed_paths,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            output_path=None,
        )
        return {
            "success": True,
            "image_base64": image_base64,
            "error_message": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "image_base64": None,
            "error_message": str(exc),
        }


def _handle_remove_background(job_input: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """Handle the 'remove_background' task_type."""
    image_path = job_input.get("image_path")
    if not image_path:
        return {
            "success": False,
            "image_base64": None,
            "error_message": "'image_path' is required for remove_background task.",
        }

    tmp_dir = os.path.join("/tmp", "bg_removed", job_id)
    os.makedirs(tmp_dir, exist_ok=True)
    output_path = os.path.join(tmp_dir, "removed.png")

    try:
        processed_path = remove_background(image_path_or_url=image_path, output_path=output_path)
        image_base64 = _encode_image_file_to_base64(processed_path)
        return {
            "success": True,
            "image_base64": image_base64,
            "error_message": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "image_base64": None,
            "error_message": str(exc),
        }


def handler(event: Dict[str, Any], context: Any | None = None) -> Dict[str, Any]:
    """Runpod serverless handler.

    Parameters
    ----------
    event:
        The RunPod job/event payload, typically:
        {
            "id": "...",
            "input": { ... }
        }
    context:
        Optional execution context for compatibility with handler(event, context)
        style signatures. Not used by this implementation.
    """
    job_id = str(event.get("id", "unknown"))
    job_input = event.get("input") or {}

    task_type = job_input.get("task_type", "infer")

    if task_type == "infer":
        return _handle_infer(job_input, job_id)
    if task_type == "remove_background":
        return _handle_remove_background(job_input, job_id)

    return {
        "success": False,
        "image_base64": None,
        "error_message": f"Unknown task_type: {task_type}",
    }


runpod.serverless.start({"handler": handler})


