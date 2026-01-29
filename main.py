"""Unified FastAPI application for inference service (includes inference and background removal)."""

from __future__ import annotations

import os

from fastapi import FastAPI

from bg_removal.models import BackgroundRemovalRequest, BackgroundRemovalResponse
from bg_removal.remover import remove_background
from infer import run_inference
from models import InferenceRequest, InferenceResponse

app = FastAPI(title="OOTD Inference Service", version="0.1.0")


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest) -> InferenceResponse:
    """
    Run inference with the given prompt and images.

    This is a pure inference service - no business logic, just model inference.
    """
    try:
        image_base64 = run_inference(
            prompt=request.prompt,
            image_paths=request.image_paths,
            height=request.height,
            width=request.width,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
        )
        return InferenceResponse(success=True, image_base64=image_base64, error_message=None)
    except Exception as exc:  # noqa: BLE001
        return InferenceResponse(success=False, image_base64=None, error_message=str(exc))


@app.post("/remove_background", response_model=BackgroundRemovalResponse)
async def remove_bg(request: BackgroundRemovalRequest) -> BackgroundRemovalResponse:
    """
    Remove background from an image.

    This is an optional standalone service. The remover can also be used as a library.
    """
    try:
        output_path = remove_background(
            image_path_or_url=request.image_path,
            output_path=request.output_path,
        )
        return BackgroundRemovalResponse(success=True, output_path=output_path, error_message=None)
    except Exception as exc:  # noqa: BLE001
        return BackgroundRemovalResponse(success=False, output_path=None, error_message=str(exc))


@app.get("/health")
async def health() -> dict:
    """Health check endpoint for unified inference service."""
    return {
        "status": "ok",
        "service": "inference",
        "services": ["inference", "background_removal"],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("INFERENCE_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)

