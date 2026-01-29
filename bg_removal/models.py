"""Request/Response models for background removal service (when used as HTTP service)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BackgroundRemovalRequest(BaseModel):
    """Request model for background removal service."""

    image_path: str = Field(..., description="Path to image (local or URL)")
    output_path: str | None = Field(default=None, description="Optional output path")


class BackgroundRemovalResponse(BaseModel):
    """Response model for background removal service."""

    success: bool = Field(..., description="Whether background removal succeeded")
    output_path: str | None = Field(default=None, description="Path to image with background removed")
    error_message: str | None = Field(default=None, description="Error message if removal failed")

