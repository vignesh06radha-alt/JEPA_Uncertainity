"""Frozen visual interventions in ImageNet-normalized image coordinates."""

import torch


LEVELS = {"tint": (0.15, 0.35, 0.60), "distractor": (0.05, 0.15, 0.30),
          "occlusion": (0.05, 0.15, 0.30)}
MODES = ("observation", "both", "goal")


def patch_shape(height, width, severity):
    return (max(1, min(height, round(height * severity ** 0.5))),
            max(1, min(width, round(width * severity ** 0.5))))


def shift_pixels(pixels, family, severity, seed=42):
    """Use a fixed top-left patch; severity is exact approximate image area.

    The patch placement is preregistered, not selected using outcomes. Whether
    it overlaps task-critical objects must be assessed from saved examples.
    """
    if family not in LEVELS or not 0 <= severity <= 1:
        raise ValueError("Unsupported visual family or severity outside [0, 1].")
    if severity == 0:
        return pixels
    if family == "tint":
        from uncertainty import tint_pixels
        return tint_pixels(pixels, severity)
    height, width = pixels.shape[-2:]
    patch_height, patch_width = patch_shape(height, width, severity)
    output = pixels.clone()
    if family == "occlusion":
        patch = pixels.new_zeros((3, patch_height, patch_width))  # ImageNet mean RGB.
    else:
        generator = torch.Generator().manual_seed(seed)
        patch = torch.rand((3, patch_height, patch_width), generator=generator).to(pixels)
        mean = pixels.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = pixels.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        patch = (patch - mean) / std
    output[..., :patch_height, :patch_width] = patch
    return output


def shift_info(info, family, severity, mode="both", seed=42):
    if mode not in MODES:
        raise ValueError("Choose observation, both, or goal visual mode.")
    output = dict(info)
    keys = ("pixels", "goal") if mode == "both" else (("pixels",) if mode == "observation" else ("goal",))
    for key in keys:
        output[key] = shift_pixels(info[key], family, severity, seed)
    return output
