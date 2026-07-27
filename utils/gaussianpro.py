"""GaussianPro anchor growth adapted to Scaffold-GS.

The implementation follows the main ideas of GaussianPro without assuming
that input images are temporally ordered:

* choose source views from a pose/frustum-overlap graph;
* render low-resolution depth maps for the reference and source views;
* propagate source depth hypotheses into the reference view;
* refine them with PatchMatch-style spatial propagation and deterministic
  depth search using multi-view patch descriptors;
* retain only hypotheses that pass multi-view depth/reprojection checks.

The accepted world-space points are inserted into Scaffold-GS by
``GaussianModel.add_propagated_anchors``.  This module deliberately contains
no optimizer mutation so its geometry routines can be tested independently.
"""

from dataclasses import dataclass
import math
import random

import torch
import torch.nn.functional as F


_INF = 1.0e9


@dataclass
class PropagationResult:
    reference_name: str
    candidate_count: int = 0
    photometric_count: int = 0
    consistent_count: int = 0
    proposed_count: int = 0
    added_count: int = 0
    mean_photo_error: float = 0.0
    mean_consistent_views: float = 0.0


def _intrinsics(camera, height, width):
    """Scale the camera's full COLMAP intrinsics to a working resolution."""
    base_width = float(camera.image_width)
    base_height = float(camera.image_height)
    scale_x = float(width) / base_width
    scale_y = float(height) / base_height
    if (
        camera.fx is not None
        and camera.fy is not None
        and camera.cx is not None
        and camera.cy is not None
    ):
        fx = float(camera.fx) * scale_x
        fy = float(camera.fy) * scale_y
        cx = (float(camera.cx) + 0.5) * scale_x - 0.5
        cy = (float(camera.cy) + 0.5) * scale_y - 0.5
    else:
        fx = width / (2.0 * math.tan(float(camera.FoVx) * 0.5))
        fy = height / (2.0 * math.tan(float(camera.FoVy) * 0.5))
        cx = (width - 1.0) * 0.5
        cy = (height - 1.0) * 0.5
    return fx, fy, cx, cy


def _pixel_grid(height, width, device, dtype):
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return u, v


def backproject_depth(depth, camera):
    """Back-project a camera-z depth map into world coordinates."""
    if depth.ndim == 3:
        depth = depth[0]
    height, width = depth.shape
    fx, fy, cx, cy = _intrinsics(camera, height, width)
    u, v = _pixel_grid(height, width, depth.device, depth.dtype)
    camera_points = torch.stack(
        ((u - cx) * depth / fx, (v - cy) * depth / fy, depth),
        dim=-1,
    )
    ones = torch.ones_like(depth).unsqueeze(-1)
    camera_h = torch.cat((camera_points, ones), dim=-1)
    camera_to_world = torch.linalg.inv(camera.world_view_transform).to(
        device=depth.device, dtype=depth.dtype
    )
    world_h = camera_h @ camera_to_world
    return world_h[..., :3] / world_h[..., 3:].clamp_min(1e-8)


def project_world(points, camera, height, width):
    """Project world points, returning pixel u/v and camera-z."""
    original_shape = points.shape[:-1]
    flat = points.reshape(-1, 3)
    ones = torch.ones(
        (flat.shape[0], 1), device=flat.device, dtype=flat.dtype
    )
    world_h = torch.cat((flat, ones), dim=-1)
    world_to_camera = camera.world_view_transform.to(
        device=flat.device, dtype=flat.dtype
    )
    camera_h = world_h @ world_to_camera
    camera_xyz = camera_h[:, :3] / camera_h[:, 3:].clamp_min(1e-8)
    z = camera_xyz[:, 2]
    fx, fy, cx, cy = _intrinsics(camera, height, width)
    safe_z = z.clamp_min(1e-8)
    u = fx * camera_xyz[:, 0] / safe_z + cx
    v = fy * camera_xyz[:, 1] / safe_z + cy
    return (
        u.reshape(original_shape),
        v.reshape(original_shape),
        z.reshape(original_shape),
    )


def _inside(u, v, z, height, width, margin=0.0):
    return (
        (z > 1e-6)
        & (u >= margin)
        & (u <= width - 1.0 - margin)
        & (v >= margin)
        & (v <= height - 1.0 - margin)
    )


def _sample_map(image, u, v, mode="bilinear"):
    """Sample BCHW/CHW data at pixel-coordinate maps."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    height, width = image.shape[-2:]
    x = 2.0 * u / max(width - 1, 1) - 1.0
    y = 2.0 * v / max(height - 1, 1) - 1.0
    grid = torch.stack((x, y), dim=-1).unsqueeze(0)
    return F.grid_sample(
        image,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)


def _patch_descriptor(image, radius):
    """Normalized local patches used by the multi-view PatchMatch cost."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if radius <= 0:
        return image.squeeze(0)
    kernel = 2 * radius + 1
    height, width = image.shape[-2:]
    gray = image.mean(dim=1, keepdim=True)
    patches = F.unfold(
        gray, kernel_size=kernel, stride=1, padding=radius
    ).reshape(1, kernel * kernel, height, width)
    patch_mean = patches.mean(dim=1, keepdim=True)
    patch_std = patches.std(dim=1, keepdim=True, unbiased=False).clamp_min(
        1e-4
    )
    normalized_patch = (patches - patch_mean) / patch_std
    colour_mean = F.avg_pool2d(
        image, kernel, stride=1, padding=radius
    )
    # NCC-like normalized structure dominates; weak mean colour resolves
    # otherwise ambiguous flat patches without overreacting to exposure.
    return torch.cat(
        (normalized_patch, 0.25 * colour_mean), dim=1
    ).squeeze(0)


def _shift_spatial(value, dy, dx):
    shifted = torch.roll(value, shifts=(dy, dx), dims=(0, 1))
    valid = torch.ones(
        value.shape[:2], device=value.device, dtype=torch.bool
    )
    if dy > 0:
        valid[:dy] = False
    elif dy < 0:
        valid[dy:] = False
    if dx > 0:
        valid[:, :dx] = False
    elif dx < 0:
        valid[:, dx:] = False
    return shifted, valid


def _depth_to_world_normal(depth, camera):
    """Finite-difference normal map for a propagated world-space depth."""
    points = backproject_depth(depth, camera)
    dx = points[:, 1:] - points[:, :-1]
    dy = points[1:] - points[:-1]
    normal = torch.cross(dy[:, :-1], dx[:-1], dim=-1)
    normal = F.normalize(normal, dim=-1, eps=1e-6)
    normal = F.pad(normal.permute(2, 0, 1), (0, 1, 0, 1))
    valid = depth > 0
    valid_inner = (
        valid[:-1, :-1]
        & valid[:-1, 1:]
        & valid[1:, :-1]
        & valid[1:, 1:]
    )
    normal_valid = torch.zeros_like(valid)
    normal_valid[:-1, :-1] = valid_inner
    return normal, normal_valid


def _world_ray_grid(camera, height, width, offset_y=0, offset_x=0):
    """World-space camera rays for a pixel-offset patch sample."""
    device = camera.world_view_transform.device
    dtype = camera.world_view_transform.dtype
    fx, fy, cx, cy = _intrinsics(camera, height, width)
    u, v = _pixel_grid(height, width, device, dtype)
    camera_points = torch.stack(
        (
            (u + float(offset_x) - cx) / fx,
            (v + float(offset_y) - cy) / fy,
            torch.ones_like(u),
        ),
        dim=-1,
    )
    camera_to_world = torch.linalg.inv(camera.world_view_transform)
    rotation = camera_to_world[:3, :3]
    return camera_points @ rotation


def _plane_propagation_candidates(depth, camera, step):
    """Propagate neighbouring slanted planes into each target pixel."""
    points = backproject_depth(depth, camera)
    normals, normal_valid = _depth_to_world_normal(depth, camera)
    normals = normals.permute(1, 2, 0)
    height, width = depth.shape
    unit_depth = torch.ones_like(depth)
    ray_point = backproject_depth(unit_depth, camera)
    centre = camera.camera_center.to(
        device=depth.device, dtype=depth.dtype
    )
    ray = ray_point - centre
    candidates = []

    for dy, dx in ((0, step), (0, -step), (step, 0), (-step, 0)):
        plane_point, shift_valid = _shift_spatial(points, dy, dx)
        plane_normal, _ = _shift_spatial(normals, dy, dx)
        source_valid, _ = _shift_spatial(
            (normal_valid & (depth > 0)).unsqueeze(-1), dy, dx
        )
        source_valid = source_valid[..., 0] & shift_valid
        numerator = (
            plane_normal * (plane_point - centre)
        ).sum(dim=-1)
        denominator = (plane_normal * ray).sum(dim=-1)
        safe_denominator = torch.where(
            denominator.abs() > 1e-6,
            denominator,
            torch.ones_like(denominator),
        )
        distance = numerator / safe_denominator
        intersection = centre + distance.unsqueeze(-1) * ray
        _, _, candidate_depth = project_world(
            intersection, camera, height, width
        )
        valid = (
            source_valid
            & (denominator.abs() > 1e-6)
            & torch.isfinite(candidate_depth)
            & (candidate_depth > 0)
        )
        candidates.append(
            torch.where(
                valid, candidate_depth, torch.zeros_like(candidate_depth)
            )
        )
    return candidates


@torch.no_grad()
def build_camera_neighbor_graph(
    cameras,
    anchor_points,
    num_neighbors=4,
    sample_count=4096,
    min_overlap=0.05,
    max_view_angle_degrees=75.0,
):
    """Build a camera graph using shared visible anchors and camera poses."""
    if len(cameras) < 2:
        return {camera.image_name: [] for camera in cameras}

    device = anchor_points.device
    count = min(int(sample_count), int(anchor_points.shape[0]))
    sample_ids = torch.linspace(
        0,
        max(anchor_points.shape[0] - 1, 0),
        count,
        device=device,
    ).long()
    samples = anchor_points.detach()[sample_ids]

    visibility = []
    centres = []
    forwards = []
    for camera in cameras:
        height, width = int(camera.image_height), int(camera.image_width)
        u, v, z = project_world(samples, camera, height, width)
        visibility.append(_inside(u, v, z, height, width))
        centres.append(camera.camera_center.detach().to(device))
        camera_to_world = torch.linalg.inv(camera.world_view_transform)
        forwards.append(
            F.normalize(camera_to_world[2, :3], dim=0).detach().to(device)
        )

    visibility = torch.stack(visibility)
    centres = torch.stack(centres)
    forwards = torch.stack(forwards)
    distance = torch.cdist(centres, centres)
    nonzero = distance[distance > 1e-8]
    distance_scale = (
        nonzero.median().clamp_min(1e-6)
        if nonzero.numel()
        else torch.tensor(1.0, device=device)
    )
    cosine_limit = math.cos(math.radians(max_view_angle_degrees))
    visible_float = visibility.float()
    common = visible_float @ visible_float.transpose(0, 1)
    visible_count = visible_float.sum(dim=1)
    overlap = common / torch.minimum(
        visible_count[:, None], visible_count[None, :]
    ).clamp_min(1.0)
    view_cosine = forwards @ forwards.transpose(0, 1)
    normalized_baseline = distance / distance_scale
    parallax_weight = (normalized_baseline / 0.25).clamp(
        min=0.05, max=1.0
    )
    proximity_weight = torch.exp(-0.15 * normalized_baseline)
    score = (
        overlap
        * (0.25 + 0.75 * view_cosine.clamp_min(0.0))
        * parallax_weight
        * proximity_weight
    )
    valid_pair = (
        (overlap >= min_overlap)
        & (view_cosine >= cosine_limit)
        & (distance > 1e-8)
    )
    score = torch.where(
        valid_pair, score, torch.full_like(score, -_INF)
    )
    graph = {}

    for ref_idx, ref_camera in enumerate(cameras):
        ranked = torch.argsort(score[ref_idx], descending=True)
        selected = [
            cameras[int(index)]
            for index in ranked[:num_neighbors]
            if score[ref_idx, index] > -_INF * 0.5
        ]
        if len(selected) < num_neighbors:
            selected_names = {camera.image_name for camera in selected}
            nearest = torch.argsort(distance[ref_idx])
            for index in nearest:
                camera = cameras[int(index)]
                if int(index) == ref_idx:
                    continue
                if camera.image_name in selected_names:
                    continue
                selected.append(camera)
                selected_names.add(camera.image_name)
                if len(selected) == num_neighbors:
                    break
        graph[ref_camera.image_name] = selected
    return graph


class GaussianProAnchorBuilder:
    """Runs PatchMatch-style propagation and proposes new Scaffold anchors."""

    def __init__(
        self,
        cameras,
        anchor_points,
        *,
        num_neighbors=4,
        graph_samples=4096,
        min_overlap=0.05,
        downsample=8,
        patch_radius=2,
        patchmatch_iterations=3,
        opacity_threshold=0.5,
        coverage_threshold=0.7,
        min_consistent_views=2,
        max_photo_error=0.35,
        reprojection_threshold=2.0,
        depth_consistency_threshold=0.03,
        normal_consistency_threshold=0.5,
        depth_discrepancy_threshold=0.15,
        max_anchors_per_step=1024,
        min_proposals_per_step=1,
        use_plane_ncc=False,
        propagate_source_views=False,
        seed=42,
    ):
        self.cameras = list(cameras)
        self.camera_by_name = {
            camera.image_name: camera for camera in self.cameras
        }
        self.graph = build_camera_neighbor_graph(
            self.cameras,
            anchor_points,
            num_neighbors=num_neighbors,
            sample_count=graph_samples,
            min_overlap=min_overlap,
        )
        self.downsample = max(1, int(downsample))
        self.patch_radius = max(0, int(patch_radius))
        self.patchmatch_iterations = max(0, int(patchmatch_iterations))
        self.opacity_threshold = float(opacity_threshold)
        self.coverage_threshold = float(coverage_threshold)
        self.min_consistent_views = max(1, int(min_consistent_views))
        self.max_photo_error = float(max_photo_error)
        self.reprojection_threshold = float(reprojection_threshold)
        self.depth_consistency_threshold = float(depth_consistency_threshold)
        self.normal_consistency_threshold = float(
            normal_consistency_threshold
        )
        self.depth_discrepancy_threshold = float(
            depth_discrepancy_threshold
        )
        self.max_anchors_per_step = max(1, int(max_anchors_per_step))
        self.min_proposals_per_step = max(1, int(min_proposals_per_step))
        self.use_plane_ncc = bool(use_plane_ncc)
        self.propagate_source_views = bool(propagate_source_views)
        self._ray_cache = {}
        self.geometry_cache = {}
        order_rng = random.Random(seed)
        self.reference_order = self.cameras.copy()
        order_rng.shuffle(self.reference_order)
        self.reference_cursor = 0

    def next_reference(self):
        camera = self.reference_order[
            self.reference_cursor % len(self.reference_order)
        ]
        self.reference_cursor += 1
        return camera

    def _render_view(
        self,
        camera,
        gaussians,
        pipe,
        background,
        render_fn,
        prefilter_fn,
    ):
        visible = prefilter_fn(camera, gaussians, pipe, background)
        package = render_fn(
            camera,
            gaussians,
            pipe,
            background,
            visible_mask=visible,
            return_depth=True,
            return_normal=True,
            return_opacity=True,
            geometry_downsample=self.downsample,
            geometry_only=True,
        )
        depth = package["render_depth"].detach()
        opacity = package["render_opacity"].detach()
        height, width = depth.shape[-2:]
        # Resize on the camera's data device first. With --data_device cpu
        # this avoids transferring a full-resolution image to VRAM merely to
        # build a 1/8-resolution propagation descriptor.
        image = F.interpolate(
            camera.original_image.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).to(device=depth.device, dtype=depth.dtype).squeeze(0)
        return {
            "camera": camera,
            "depth": depth,
            "normal": package["render_normal"].detach(),
            "opacity": opacity,
            "image": image,
            "descriptor": _patch_descriptor(image, self.patch_radius),
        }

    def _splat_source_depth(self, source, reference):
        source_depth = source["depth"][0]
        source_valid = (
            torch.isfinite(source_depth)
            & (source_depth > 0)
            & (source["opacity"][0] >= self.opacity_threshold)
        )
        source_world = backproject_depth(
            source_depth, source["camera"]
        )[source_valid]
        ref_height, ref_width = reference["depth"].shape[-2:]
        u, v, z = project_world(
            source_world, reference["camera"], ref_height, ref_width
        )
        inside = _inside(u, v, z, ref_height, ref_width)
        if not inside.any():
            return torch.zeros(
                (ref_height, ref_width),
                device=source_depth.device,
                dtype=source_depth.dtype,
            )
        u = u[inside].round().long()
        v = v[inside].round().long()
        z = z[inside]
        flat_index = v * ref_width + u
        splatted = torch.full(
            (ref_height * ref_width,),
            _INF,
            device=z.device,
            dtype=z.dtype,
        )
        splatted.scatter_reduce_(
            0, flat_index, z, reduce="amin", include_self=True
        )
        return torch.where(
            splatted < _INF,
            splatted,
            torch.zeros_like(splatted),
        ).reshape(ref_height, ref_width)

    def _photometric_cost(self, candidate_depth, reference, sources):
        if self.use_plane_ncc:
            return self._photometric_cost_plane_ncc(
                candidate_depth, reference, sources
            )

        ref_height, ref_width = candidate_depth.shape
        valid_depth = torch.isfinite(candidate_depth) & (candidate_depth > 0)
        world = backproject_depth(candidate_depth, reference["camera"])
        cost_sum = torch.zeros_like(candidate_depth)
        view_count = torch.zeros_like(candidate_depth)
        ref_descriptor = reference["descriptor"]

        for source in sources:
            src_height, src_width = source["depth"].shape[-2:]
            u, v, z = project_world(
                world, source["camera"], src_height, src_width
            )
            valid = valid_depth & _inside(
                u,
                v,
                z,
                src_height,
                src_width,
                margin=float(self.patch_radius),
            )
            sampled_opacity = _sample_map(
                source["opacity"], u, v, mode="bilinear"
            )[0]
            valid &= sampled_opacity >= self.opacity_threshold
            sampled_descriptor = _sample_map(
                source["descriptor"], u, v, mode="bilinear"
            )
            error = (sampled_descriptor - ref_descriptor).abs().mean(dim=0)
            cost_sum += torch.where(valid, error, torch.zeros_like(error))
            view_count += valid.to(view_count.dtype)

        cost = cost_sum / view_count.clamp_min(1.0)
        cost = torch.where(
            view_count >= self.min_consistent_views,
            cost,
            torch.full_like(cost, _INF),
        )
        return cost, view_count

    def _reference_ray(self, camera, height, width, dy, dx):
        key = (
            camera.image_name,
            int(height),
            int(width),
            int(dy),
            int(dx),
        )
        if key not in self._ray_cache:
            self._ray_cache[key] = _world_ray_grid(
                camera, height, width, dy, dx
            )
        return self._ray_cache[key]

    def _photometric_cost_plane_ncc(
        self, candidate_depth, reference, sources
    ):
        """Plane-induced multi-view warp scored with normalized correlation."""
        height, width = candidate_depth.shape
        camera = reference["camera"]
        plane_point = backproject_depth(candidate_depth, camera)
        plane_normal, normal_valid = _depth_to_world_normal(
            candidate_depth, camera
        )
        plane_normal = plane_normal.permute(1, 2, 0)
        centre = camera.camera_center.to(
            device=candidate_depth.device, dtype=candidate_depth.dtype
        )
        plane_numerator = (
            plane_normal * (plane_point - centre)
        ).sum(dim=-1)
        ref_gray = reference["image"].mean(dim=0)
        radius = max(1, self.patch_radius)
        offsets = [
            (-radius, -radius),
            (-radius, 0),
            (-radius, radius),
            (0, -radius),
            (0, 0),
            (0, radius),
            (radius, -radius),
            (radius, 0),
            (radius, radius),
        ]
        total_cost = torch.zeros_like(candidate_depth)
        view_count = torch.zeros_like(candidate_depth)

        for source in sources:
            sum_x = torch.zeros_like(candidate_depth)
            sum_y = torch.zeros_like(candidate_depth)
            sum_x2 = torch.zeros_like(candidate_depth)
            sum_y2 = torch.zeros_like(candidate_depth)
            sum_xy = torch.zeros_like(candidate_depth)
            sample_count = torch.zeros_like(candidate_depth)
            source_gray = source["image"].mean(dim=0, keepdim=True)
            src_height, src_width = source["depth"].shape[-2:]

            for dy, dx in offsets:
                ray = self._reference_ray(
                    camera, height, width, dy, dx
                )
                denominator = (plane_normal * ray).sum(dim=-1)
                safe_denominator = torch.where(
                    denominator.abs() > 1e-6,
                    denominator,
                    torch.ones_like(denominator),
                )
                distance = plane_numerator / safe_denominator
                warped_world = centre + distance.unsqueeze(-1) * ray
                src_u, src_v, src_z = project_world(
                    warped_world,
                    source["camera"],
                    src_height,
                    src_width,
                )
                ref_sample, ref_valid = _shift_spatial(
                    ref_gray.unsqueeze(-1), -dy, -dx
                )
                ref_sample = ref_sample[..., 0]
                valid = (
                    normal_valid
                    & (candidate_depth > 0)
                    & ref_valid
                    & (denominator.abs() > 1e-6)
                    & (distance > 0)
                    & _inside(
                        src_u,
                        src_v,
                        src_z,
                        src_height,
                        src_width,
                        margin=1.0,
                    )
                )
                src_sample = _sample_map(
                    source_gray, src_u, src_v, mode="bilinear"
                )[0]
                weight = valid.to(candidate_depth.dtype)
                sum_x += weight * ref_sample
                sum_y += weight * src_sample
                sum_x2 += weight * ref_sample.square()
                sum_y2 += weight * src_sample.square()
                sum_xy += weight * ref_sample * src_sample
                sample_count += weight

            count = sample_count.clamp_min(1.0)
            covariance = sum_xy - sum_x * sum_y / count
            variance_x = (sum_x2 - sum_x.square() / count).clamp_min(
                1e-6
            )
            variance_y = (sum_y2 - sum_y.square() / count).clamp_min(
                1e-6
            )
            ncc = covariance / torch.sqrt(variance_x * variance_y)
            ncc_cost = 0.5 * (1.0 - ncc.clamp(-1.0, 1.0))
            valid_view = sample_count >= max(5, len(offsets) - 2)
            total_cost += torch.where(
                valid_view, ncc_cost, torch.zeros_like(ncc_cost)
            )
            view_count += valid_view.to(view_count.dtype)

        cost = total_cost / view_count.clamp_min(1.0)
        cost = torch.where(
            view_count >= self.min_consistent_views,
            cost,
            torch.full_like(cost, _INF),
        )
        return cost, view_count

    def _select_best(self, depth_candidates, reference, sources):
        costs = []
        counts = []
        for candidate in depth_candidates:
            cost, count = self._photometric_cost(
                candidate, reference, sources
            )
            costs.append(cost)
            counts.append(count)
        cost_stack = torch.stack(costs)
        best_cost, best_index = cost_stack.min(dim=0)
        depth_stack = torch.stack(depth_candidates)
        best_depth = torch.gather(
            depth_stack, 0, best_index.unsqueeze(0)
        ).squeeze(0)
        count_stack = torch.stack(counts)
        best_count = torch.gather(
            count_stack, 0, best_index.unsqueeze(0)
        ).squeeze(0)
        return best_depth, best_cost, best_count

    def _solve_view(self, reference, sources):
        reference_depth = reference["depth"][0]
        candidates = [
            torch.where(
                reference["opacity"][0] >= self.opacity_threshold,
                reference_depth,
                torch.zeros_like(reference_depth),
            )
        ]
        candidates.extend(
            self._splat_source_depth(source, reference)
            for source in sources
        )
        best_depth, best_cost, best_views = self._select_best(
            candidates, reference, sources
        )
        for patchmatch_iteration in range(self.patchmatch_iterations):
            step = min(2 ** patchmatch_iteration, 8)
            search = 0.08 / (patchmatch_iteration + 1)
            refinements = [
                best_depth,
                *_plane_propagation_candidates(
                    best_depth, reference["camera"], step
                ),
                best_depth * (1.0 - search),
                best_depth * (1.0 + search),
            ]
            best_depth, best_cost, best_views = self._select_best(
                refinements, reference, sources
            )
        return best_depth, best_cost, best_views, candidates

    def _cache_view_geometry(
        self,
        view,
        depth,
        normal_camera,
        valid_mask,
        attach_supervision=False,
    ):
        cached = {
            "depth": torch.where(
                valid_mask, depth, torch.zeros_like(depth)
            ).unsqueeze(0).detach().cpu(),
            "normal": torch.where(
                valid_mask.unsqueeze(0),
                normal_camera,
                torch.zeros_like(normal_camera),
            ).detach().cpu(),
            "valid": valid_mask.detach().cpu(),
        }
        self.geometry_cache[view["camera"].image_name] = cached
        if attach_supervision and valid_mask.any():
            camera = view["camera"]
            camera.gaussianpro_depth_target = cached["depth"]
            camera.gaussianpro_normal_target = cached["normal"]
            camera.gaussianpro_target_mask = cached["valid"]

    def _apply_cached_geometry(self, view):
        cached = self.geometry_cache.get(view["camera"].image_name)
        if cached is None:
            return False
        device = view["depth"].device
        dtype = view["depth"].dtype
        view["depth"] = cached["depth"].to(device=device, dtype=dtype)
        view["normal"] = cached["normal"].to(device=device, dtype=dtype)
        valid = cached["valid"].to(device=device)
        view["opacity"] = valid.to(dtype).unsqueeze(0)
        return True

    def _normal_world_to_camera(self, normal_world, camera):
        normal_camera = (
            normal_world.permute(1, 2, 0)
            @ camera.world_view_transform[:3, :3]
        )
        return F.normalize(
            normal_camera, dim=-1, eps=1e-6
        ).permute(2, 0, 1)

    def _geometric_consistency(self, depth, reference, sources):
        ref_height, ref_width = depth.shape
        ref_u, ref_v = _pixel_grid(
            ref_height, ref_width, depth.device, depth.dtype
        )
        world = backproject_depth(depth, reference["camera"])
        reference_normal, reference_normal_valid = _depth_to_world_normal(
            depth, reference["camera"]
        )
        reference_normal = reference_normal.permute(1, 2, 0)
        consistent = torch.zeros_like(depth)

        for source in sources:
            src_height, src_width = source["depth"].shape[-2:]
            src_u, src_v, projected_z = project_world(
                world, source["camera"], src_height, src_width
            )
            valid = (depth > 0) & _inside(
                src_u, src_v, projected_z, src_height, src_width
            )
            sampled_depth = _sample_map(
                source["depth"], src_u, src_v, mode="nearest"
            )[0]
            sampled_opacity = _sample_map(
                source["opacity"], src_u, src_v, mode="nearest"
            )[0]
            sampled_normal_camera = _sample_map(
                source["normal"], src_u, src_v, mode="bilinear"
            ).permute(1, 2, 0)
            camera_to_world_rotation = torch.linalg.inv(
                source["camera"].world_view_transform[:3, :3]
            ).to(device=depth.device, dtype=depth.dtype)
            sampled_normal_world = F.normalize(
                sampled_normal_camera @ camera_to_world_rotation,
                dim=-1,
                eps=1e-6,
            )
            normal_agreement = (
                sampled_normal_world * reference_normal
            ).sum(dim=-1).abs()
            relative_depth = (
                (sampled_depth - projected_z).abs()
                / sampled_depth.clamp_min(1e-6)
            )

            fx, fy, cx, cy = _intrinsics(
                source["camera"], src_height, src_width
            )
            sampled_u = src_u.round()
            sampled_v = src_v.round()
            source_points = torch.stack(
                (
                    (sampled_u - cx) * sampled_depth / fx,
                    (sampled_v - cy) * sampled_depth / fy,
                    sampled_depth,
                ),
                dim=-1,
            )
            ones = torch.ones_like(sampled_depth).unsqueeze(-1)
            source_h = torch.cat((source_points, ones), dim=-1)
            source_to_world = torch.linalg.inv(
                source["camera"].world_view_transform
            ).to(device=depth.device, dtype=depth.dtype)
            reprojected_world = source_h @ source_to_world
            reprojected_world = (
                reprojected_world[..., :3]
                / reprojected_world[..., 3:].clamp_min(1e-8)
            )
            check_u, check_v, check_z = project_world(
                reprojected_world,
                reference["camera"],
                ref_height,
                ref_width,
            )
            reprojection = torch.sqrt(
                (check_u - ref_u).square() + (check_v - ref_v).square()
            )
            valid &= (
                (sampled_depth > 0)
                & (sampled_opacity >= self.opacity_threshold)
                & reference_normal_valid
                & (
                    normal_agreement
                    >= self.normal_consistency_threshold
                )
                & (check_z > 0)
                & (
                    relative_depth
                    <= self.depth_consistency_threshold
                )
                & (reprojection <= self.reprojection_threshold)
            )
            consistent += valid.to(consistent.dtype)
        return consistent

    @torch.no_grad()
    def run(
        self,
        reference_camera,
        gaussians,
        pipe,
        background,
        render_fn,
        prefilter_fn,
    ):
        result = PropagationResult(reference_camera.image_name)
        self._ray_cache.clear()
        neighbor_cameras = self.graph.get(reference_camera.image_name, [])
        if len(neighbor_cameras) < self.min_consistent_views:
            return result, None, None

        rendered_views = {}

        def get_rendered(camera):
            if camera.image_name not in rendered_views:
                rendered_views[camera.image_name] = self._render_view(
                    camera,
                    gaussians,
                    pipe,
                    background,
                    render_fn,
                    prefilter_fn,
                )
            return rendered_views[camera.image_name]

        reference = get_rendered(reference_camera)
        sources = [get_rendered(camera) for camera in neighbor_cameras]

        if self.propagate_source_views:
            for source in sources:
                if self._apply_cached_geometry(source):
                    continue
                support_cameras = self.graph.get(
                    source["camera"].image_name, []
                )
                support_views = [
                    get_rendered(camera)
                    for camera in support_cameras
                    if camera.image_name
                    != source["camera"].image_name
                ]
                if len(support_views) < self.min_consistent_views:
                    continue
                for support in support_views:
                    self._apply_cached_geometry(support)
                self._ray_cache.clear()
                (
                    source_depth,
                    source_cost,
                    source_views,
                    _,
                ) = self._solve_view(source, support_views)
                source_valid = (
                    torch.isfinite(source_cost)
                    & (source_cost <= self.max_photo_error)
                    & (
                        source_views
                        >= self.min_consistent_views
                    )
                    & (source_depth > 0)
                )
                source_consistent_views = self._geometric_consistency(
                    source_depth, source, support_views
                )
                source_valid &= (
                    source_consistent_views
                    >= self.min_consistent_views
                )
                source_normal_world, source_normal_valid = (
                    _depth_to_world_normal(
                        source_depth, source["camera"]
                    )
                )
                source_valid &= source_normal_valid
                source_normal_camera = (
                    self._normal_world_to_camera(
                        source_normal_world, source["camera"]
                    )
                )
                self._cache_view_geometry(
                    source,
                    source_depth,
                    source_normal_camera,
                    source_valid,
                    attach_supervision=True,
                )
                self._apply_cached_geometry(source)

        self._ray_cache.clear()
        (
            best_depth,
            best_cost,
            best_views,
            candidates,
        ) = self._solve_view(reference, sources)

        reference_depth = reference["depth"][0]
        result.candidate_count = int(
            torch.stack([candidate > 0 for candidate in candidates])
            .any(dim=0)
            .sum()
            .item()
        )

        photo_mask = (
            torch.isfinite(best_cost)
            & (best_cost <= self.max_photo_error)
            & (best_views >= self.min_consistent_views)
            & (best_depth > 0)
        )
        result.photometric_count = int(photo_mask.sum().item())
        consistent_views = self._geometric_consistency(
            best_depth, reference, sources
        )
        geometry_mask = consistent_views >= self.min_consistent_views
        result.consistent_count = int((photo_mask & geometry_mask).sum().item())
        propagated_normal_world, normal_valid = _depth_to_world_normal(
            best_depth, reference_camera
        )
        propagated_normal_camera = self._normal_world_to_camera(
            propagated_normal_world, reference_camera
        )
        target_valid = photo_mask & geometry_mask & normal_valid
        self._cache_view_geometry(
            reference,
            best_depth,
            propagated_normal_camera,
            target_valid,
            attach_supervision=True,
        )

        current_valid = reference_depth > 0
        relative_difference = torch.where(
            current_valid,
            (best_depth - reference_depth).abs()
            / best_depth.clamp_min(1e-6),
            torch.full_like(best_depth, _INF),
        )
        under_covered = (
            reference["opacity"][0] < self.coverage_threshold
        )
        needs_anchor = (
            under_covered
            | (relative_difference >= self.depth_discrepancy_threshold)
        )
        border_valid = torch.ones_like(photo_mask)
        if self.patch_radius > 0:
            border_valid[: self.patch_radius] = False
            border_valid[-self.patch_radius :] = False
            border_valid[:, : self.patch_radius] = False
            border_valid[:, -self.patch_radius :] = False
        accepted = (
            photo_mask
            & geometry_mask
            & needs_anchor
            & normal_valid
            & border_valid
        )

        accepted_indices = accepted.flatten().nonzero().squeeze(1)
        result.proposed_count = int(accepted_indices.numel())
        if accepted_indices.numel() < self.min_proposals_per_step:
            return result, None, None

        score = (
            consistent_views
            + (1.0 - best_cost.clamp(0.0, 1.0))
            + relative_difference.clamp(0.0, 2.0)
            + (1.0 - reference["opacity"][0])
        ).flatten()[accepted_indices]
        if accepted_indices.numel() > self.max_anchors_per_step:
            keep = torch.topk(
                score, self.max_anchors_per_step, sorted=False
            ).indices
            accepted_indices = accepted_indices[keep]

        world = backproject_depth(best_depth, reference_camera)
        points = world.reshape(-1, 3)[accepted_indices]
        point_normals = (
            propagated_normal_world.permute(1, 2, 0)
            .reshape(-1, 3)[accepted_indices]
        )
        result.proposed_count = int(points.shape[0])
        if points.numel():
            result.mean_photo_error = float(
                best_cost.flatten()[accepted_indices].mean().item()
            )
            result.mean_consistent_views = float(
                consistent_views.flatten()[accepted_indices].mean().item()
            )
        return result, points, point_normals
