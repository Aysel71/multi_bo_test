"""HPSv3 reward model wrapped for MultiBO non-human scoring.

`non_human_choice` (objectives/low_dims/ImageGen/*.py) calls
``Selector(device).score([pil_image], prompt)[0]`` for every candidate, so
this exposes the same ``Selector.score(images, prompt) -> list[float]``
interface as pickscore_utils / hps_utils.

HPSv3 scores images from file paths, not PIL objects, so each PIL image is
written to a temporary PNG before scoring. The heavy ``HPSv3RewardInferencer``
is loaded once and cached at module level — ``non_human_choice`` builds a fresh
Selector on every BO batch, and reloading a multi-GB model each time would be
prohibitively slow.

Port of method A's scorer (noise_bo_flux/rewards/hpsv3.py): same reward()
API-variant probing, same scalar extraction.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import torch
from PIL import Image

log = logging.getLogger(__name__)

# Loaded once, reused across every Selector instance / BO batch.
_INFERENCER: Any = None
_API_VARIANT: str | None = None


def _extract_first_scalar(obj: Any) -> float:
    """Pull a single float out of whatever shape HPSv3 returns."""
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        flat = obj.reshape(-1)
        if flat.numel() == 0:
            raise ValueError("Empty tensor in HPSv3 result")
        return float(flat[0].item())
    if hasattr(obj, "ndim") and hasattr(obj, "flatten"):
        flat = obj.flatten()
        if len(flat) == 0:
            raise ValueError("Empty array in HPSv3 result")
        return float(flat[0])
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            raise ValueError("Empty sequence in HPSv3 result")
        return _extract_first_scalar(obj[0])
    if hasattr(obj, "item"):
        try:
            return float(obj.item())
        except (ValueError, RuntimeError):
            pass
    raise TypeError(
        f"Cannot extract scalar from HPSv3 result of type {type(obj)}: {obj!r}"
    )


class Selector:
    """HPSv3 scorer. ``score(images, prompt)`` returns a list of float rewards."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        global _INFERENCER
        if _INFERENCER is None:
            from hpsv3 import HPSv3RewardInferencer  # type: ignore[import-not-found]
            log.info("Loading HPSv3 reward model (one-time)")
            _INFERENCER = HPSv3RewardInferencer(device=device)
        self.model = _INFERENCER

    def _call_reward(self, image_path: str, prompt: str):
        """Call HPSv3 reward(), probing the API variant once and caching it."""
        global _API_VARIANT

        if _API_VARIANT in (None, "two_lists"):
            try:
                res = self.model.reward([image_path], [prompt])
                _API_VARIANT = "two_lists"
                return res
            except TypeError as e:
                if _API_VARIANT == "two_lists":
                    raise
                log.debug("HPSv3 two_lists failed: %s, trying dict_list", e)

        if _API_VARIANT in (None, "dict_list"):
            try:
                res = self.model.reward([{"image_path": [image_path], "prompt": prompt}])
                _API_VARIANT = "dict_list"
                return res
            except TypeError:
                if _API_VARIANT == "dict_list":
                    raise

        try:
            res = self.model.reward(image_paths=[image_path], prompts=[prompt])
            _API_VARIANT = "kwargs"
            return res
        except TypeError as e:
            raise RuntimeError(
                f"All known HPSv3 reward() API variants failed. Last: {e}"
            ) from e

    def score(self, images, prompt):
        """images: list of PIL.Image (or file paths). Returns list[float]."""
        if not isinstance(images, (list, tuple)):
            images = [images]

        scores: list[float] = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, img in enumerate(images):
                if isinstance(img, Image.Image):
                    path = os.path.join(tmp, f"cand_{i}.png")
                    img.convert("RGB").save(path)
                else:
                    path = str(img)
                with torch.no_grad():
                    result = self._call_reward(path, prompt)
                scores.append(_extract_first_scalar(result))
        return scores
