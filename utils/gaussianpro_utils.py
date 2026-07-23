import math

import torch
import torch.nn.functional as F


def _resize_like(image, reference):
    if image.shape[-2:] == reference.shape[-2:]:
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _depth_to_camera_normal(depth, opacity, fov_x, fov_y, opacity_threshold):
    """Convert a camera-z map to finite-difference camera-space normals."""
    height, width = depth.shape[-2:]
    raw_z = depth[0]
    z = torch.nan_to_num(raw_z, nan=0.0, posinf=0.0, neginf=0.0)
    device, dtype = z.device, z.dtype

    fx = width / (2.0 * math.tan(float(fov_x) * 0.5))
    fy = height / (2.0 * math.tan(float(fov_y) * 0.5))
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    points = torch.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), dim=-1)

    dx = points[:, 1:] - points[:, :-1]
    dy = points[1:] - points[:-1]
    # Image y points down. dy x dx therefore faces the camera (-camera z)
    # for a fronto-parallel surface, matching the rendered Gaussian normals.
    normal = torch.cross(dy[:, :-1], dx[:-1], dim=-1)
    normal = F.normalize(normal, dim=-1, eps=1e-6)

    finite = torch.isfinite(raw_z) & (raw_z > 0)
    opaque = opacity[0] > opacity_threshold
    valid_pixel = finite & opaque
    valid_inner = (
        valid_pixel[:-1, :-1]
        & valid_pixel[:-1, 1:]
        & valid_pixel[1:, :-1]
        & valid_pixel[1:, 1:]
    )
    normal = F.pad(normal.permute(2, 0, 1), (0, 1, 0, 1))
    valid = torch.zeros_like(valid_pixel)
    valid[:-1, :-1] = valid_inner
    return normal, valid


def _weighted_mean(value, weight):
    weight_sum = weight.sum()
    return (value * weight).sum() / weight_sum.clamp_min(1e-6)


def gaussianpro_geometry_loss(
    rendered_depth,
    rendered_normal,
    rendered_opacity,
    gt_image,
    fov_x,
    fov_y,
    opacity_threshold=0.5,
    edge_weight=10.0,
):
    """Geometry losses adapted from GaussianPro for Scaffold-GS anchors.

    The depth-derived normal is detached and acts as a local planar target for
    the covariance normal. The second-order inverse-depth term is scale
    invariant and is suppressed at RGB edges, so real object boundaries are
    not intentionally smoothed over.
    """
    gt_image = _resize_like(gt_image, rendered_depth).detach()
    rendered_normal = _resize_like(rendered_normal, rendered_depth)
    rendered_opacity = _resize_like(rendered_opacity, rendered_depth)

    with torch.no_grad():
        target_normal, normal_valid = _depth_to_camera_normal(
            rendered_depth.detach(),
            rendered_opacity.detach(),
            fov_x,
            fov_y,
            opacity_threshold,
        )
        gray = gt_image.mean(dim=0)
        gx = F.pad((gray[:, 1:] - gray[:, :-1]).abs(), (0, 1, 0, 0))
        gy = F.pad((gray[1:] - gray[:-1]).abs(), (0, 0, 0, 1))
        edge_confidence = torch.exp(-edge_weight * (gx + gy))

    predicted_normal = F.normalize(rendered_normal, dim=0, eps=1e-6)
    cosine_error = 1.0 - (predicted_normal * target_normal).sum(dim=0).clamp(-1.0, 1.0)
    l1_error = (predicted_normal - target_normal).abs().mean(dim=0)
    normal_weight = normal_valid.to(rendered_depth.dtype) * edge_confidence
    normal_loss = _weighted_mean(
        0.5 * cosine_error + 0.5 * l1_error,
        normal_weight,
    )

    valid_depth = (
        torch.isfinite(rendered_depth[0])
        & (rendered_depth[0] > 0)
        & (rendered_opacity[0] > opacity_threshold)
    )
    safe_depth = torch.nan_to_num(
        rendered_depth[0], nan=0.0, posinf=0.0, neginf=0.0
    )
    inverse_depth = safe_depth.clamp_min(1e-6).reciprocal()
    valid_float = valid_depth.to(inverse_depth.dtype)
    depth_scale = (
        (inverse_depth * valid_float).sum()
        / valid_float.sum().clamp_min(1.0)
    ).detach().clamp_min(1e-6)
    inverse_depth = inverse_depth / depth_scale

    dxx = inverse_depth[:, 2:] - 2.0 * inverse_depth[:, 1:-1] + inverse_depth[:, :-2]
    dyy = inverse_depth[2:] - 2.0 * inverse_depth[1:-1] + inverse_depth[:-2]
    valid_x = valid_depth[:, 2:] & valid_depth[:, 1:-1] & valid_depth[:, :-2]
    valid_y = valid_depth[2:] & valid_depth[1:-1] & valid_depth[:-2]
    weight_x = torch.exp(-edge_weight * (gray[:, 2:] - gray[:, :-2]).abs())
    weight_y = torch.exp(-edge_weight * (gray[2:] - gray[:-2]).abs())
    smooth_x = _weighted_mean(dxx.abs(), weight_x * valid_x.to(weight_x.dtype))
    smooth_y = _weighted_mean(dyy.abs(), weight_y * valid_y.to(weight_y.dtype))
    depth_smooth_loss = 0.5 * (smooth_x + smooth_y)

    valid_ratio = normal_valid.float().mean()
    return normal_loss, depth_smooth_loss, valid_ratio
