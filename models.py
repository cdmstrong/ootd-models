"""Request/Response models for inference service."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Request model for inference service."""

    prompt: str = Field(..., description="Text prompt for image generation")
    image_paths: List[str] = Field(
        ...,
        description="List of image paths (local or URLs). First image is the person/base image, followed by accessory images.",
        min_items=1,
        max_items=4,
    )
    height: int = Field(default=1024, description="Output image height")
    width: int = Field(default=1024, description="Output image width")
    guidance_scale: float = Field(default=1.0, description="Guidance scale for generation")
    num_inference_steps: int = Field(default=10, description="Number of inference steps")


class InferenceResponse(BaseModel):
    """Response model for inference service."""

    success: bool = Field(..., description="Whether inference succeeded")
    image_base64: Optional[str] = Field(default=None, description="Base64-encoded generated image (PNG)")
    error_message: Optional[str] = Field(default=None, description="Error message if inference failed")

