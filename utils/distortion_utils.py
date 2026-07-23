import torch
import torch.nn.functional as F


_DISTORTION_GRID_CACHE = {}


def _redistortion_grid(camera, height, width, device, dtype):
    key = (
        height,
        width,
        float(camera.fx),
        float(camera.fy),
        float(camera.cx),
        float(camera.cy),
        float(camera.radial_k1),
        device.type,
        device.index,
        dtype,
    )
    cached = _DISTORTION_GRID_CACHE.get(key)
    if cached is not None:
        return cached

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    distorted_x = (x - camera.cx) / camera.fx
    distorted_y = (y - camera.cy) / camera.fy

    # Invert COLMAP SIMPLE_RADIAL:
    # x_distorted = x_undistorted * (1 + k1 * r_undistorted^2).
    undistorted_x = distorted_x.clone()
    undistorted_y = distorted_y.clone()
    for _ in range(8):
        radius_squared = (
            undistorted_x.square() + undistorted_y.square()
        )
        radial = 1.0 + camera.radial_k1 * radius_squared
        undistorted_x = distorted_x / radial.clamp_min(1e-8)
        undistorted_y = distorted_y / radial.clamp_min(1e-8)

    source_x = camera.fx * undistorted_x + camera.cx
    source_y = camera.fy * undistorted_y + camera.cy
    grid_x = 2.0 * source_x / max(width - 1, 1) - 1.0
    grid_y = 2.0 * source_y / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    _DISTORTION_GRID_CACHE[key] = grid
    return grid


def distort_render_to_raw(image, camera):
    """Warp a pinhole/undistorted CHW tensor into the raw radial camera domain."""
    if (
        not getattr(camera, "image_undistorted", False)
        or abs(getattr(camera, "radial_k1", 0.0)) <= 1e-12
    ):
        return image
    if any(
        getattr(camera, name, None) is None
        for name in ("fx", "fy", "cx", "cy")
    ):
        raise ValueError("Camera intrinsics are required for radial redistortion")

    height, width = image.shape[-2:]
    grid = _redistortion_grid(
        camera,
        height,
        width,
        image.device,
        image.dtype,
    )
    return F.grid_sample(
        image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
