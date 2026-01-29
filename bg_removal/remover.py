"""Background removal logic using rembg."""

from __future__ import annotations

import os
from io import BytesIO
from urllib.parse import urlparse

import requests
from PIL import Image
from rembg import new_session, remove

# Global session to cache the model (loaded once, reused for all requests)
_BG_REMOVAL_SESSION = None


def _get_session():
    """Get or create the rembg session (singleton pattern)."""
    global _BG_REMOVAL_SESSION
    if _BG_REMOVAL_SESSION is None:
        _BG_REMOVAL_SESSION = new_session()
    return _BG_REMOVAL_SESSION


def remove_background(
    image_path_or_url: str,
    output_path: str | None = None,
    output_dir: str = "outputs/bg_removed",
) -> str:
    """
    Remove background from an image.

    Args:
        image_path_or_url: Path to local image or URL
        output_path: Optional output path. If None, generates a path in output_dir.
        output_dir: Directory to save output if output_path is None.

    Returns:
        Path to the image with background removed.
    """
    # Load image
    parsed = urlparse(image_path_or_url)
    if parsed.scheme in ("http", "https"):
        # Download from URL
        response = requests.get(image_path_or_url, timeout=30)
        response.raise_for_status()
        input_image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        # Local file path
        input_image = Image.open(image_path_or_url).convert("RGB")

    # Remove background (use cached session for better performance)
    session = _get_session()
    output_image = remove(input_image, session=session)

    # Determine output path
    if output_path is None:
        os.makedirs(output_dir, exist_ok=True)
        # Generate filename from input
        if parsed.scheme in ("http", "https"):
            # Use URL path as base for filename
            base_name = os.path.basename(parsed.path) or "image"
            if "." not in base_name:
                base_name += ".png"
        else:
            base_name = os.path.basename(image_path_or_url)
            if "." not in base_name:
                base_name += ".png"

        # Ensure .png extension
        if not base_name.endswith(".png"):
            base_name = os.path.splitext(base_name)[0] + ".png"

        output_path = os.path.join(output_dir, base_name)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Save result
    output_image.save(output_path)
    return output_path

