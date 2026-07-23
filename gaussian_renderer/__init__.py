#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import torch.nn.functional as F
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.general_utils import build_rotation

def get_embedding_num_cameras(appearance_module):
    if appearance_module is None:
        return 0
    if hasattr(appearance_module, 'in_dim'):
        return appearance_module.in_dim
    try:
        if hasattr(appearance_module, 'embedding') and hasattr(appearance_module.embedding, 'weight'):
            return appearance_module.embedding.weight.shape[0]
    except Exception:
        pass
    try:
        for param in appearance_module.parameters():
            return param.shape[0]
    except Exception:
        pass
    return 999999

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        # Clamp to avoid index out of bounds for test poses
        num_cams = get_embedding_num_cameras(pc.embedding_appearance)
        cam_idx = min(viewpoint_camera.uid, num_cams - 1) if num_cams > 0 else viewpoint_camera.uid
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * cam_idx
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,
           scaling_modifier=1.0, visible_mask=None, retain_grad=False,
           return_depth=False, return_normal=False, return_opacity=False,
           geometry_downsample=1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)

    geometry_outputs = {}
    if return_depth or return_normal or return_opacity:
        geometry_downsample = max(1, int(geometry_downsample))
        geometry_height = max(1, int(viewpoint_camera.image_height) // geometry_downsample)
        geometry_width = max(1, int(viewpoint_camera.image_width) // geometry_downsample)
        geometry_settings = GaussianRasterizationSettings(
            image_height=geometry_height,
            image_width=geometry_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.zeros_like(bg_color),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        geometry_rasterizer = GaussianRasterizer(raster_settings=geometry_settings)

        # Pack accumulated depth and alpha into one RGB raster pass:
        # channel 0 = sum(T_i * alpha_i * z_i), channel 1 = sum(T_i * alpha_i).
        if return_depth or return_opacity:
            viewmatrix = viewpoint_camera.world_view_transform
            camera_depth = (
                (xyz * viewmatrix[:3, 2].unsqueeze(0)).sum(dim=-1, keepdim=True)
                + viewmatrix[3, 2]
            )
            depth_alpha_features = torch.cat(
                [camera_depth, torch.ones_like(camera_depth), torch.zeros_like(camera_depth)],
                dim=-1
            )
            depth_alpha, _ = geometry_rasterizer(
                means3D=xyz,
                means2D=screenspace_points,
                shs=None,
                colors_precomp=depth_alpha_features,
                opacities=opacity,
                scales=scaling,
                rotations=rot,
                cov3D_precomp=None
            )
            rendered_opacity = depth_alpha[1:2].clamp(0.0, 1.0)
            if return_opacity:
                geometry_outputs["render_opacity"] = rendered_opacity
            if return_depth:
                valid_alpha = rendered_opacity > 1e-6
                rendered_depth = depth_alpha[0:1] / rendered_opacity.clamp_min(1e-6)
                geometry_outputs["render_depth"] = torch.where(
                    valid_alpha, rendered_depth, torch.zeros_like(rendered_depth)
                )

        if return_normal:
            rotation_matrices = build_rotation(rot)
            min_axis = scaling.argmin(dim=-1)
            gaussian_indices = torch.arange(xyz.shape[0], device=xyz.device)
            normal_world = rotation_matrices[gaussian_indices, :, min_axis]

            # A Gaussian plane has two equivalent normal directions. Orient it
            # towards the current camera before blending.
            camera_to_gaussian = xyz - viewpoint_camera.camera_center.unsqueeze(0)
            facing_away = (normal_world * camera_to_gaussian).sum(dim=-1, keepdim=True) > 0
            normal_world = torch.where(facing_away, -normal_world, normal_world)
            normal_camera = normal_world @ viewpoint_camera.world_view_transform[:3, :3]
            normal_camera = F.normalize(normal_camera, dim=-1, eps=1e-6)

            rendered_normal, _ = geometry_rasterizer(
                means3D=xyz,
                means2D=screenspace_points,
                shs=None,
                colors_precomp=normal_camera,
                opacities=opacity,
                scales=scaling,
                rotations=rot,
                cov3D_precomp=None
            )
            geometry_outputs["render_normal"] = F.normalize(
                rendered_normal, dim=0, eps=1e-6
            )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        result = {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
        result.update(geometry_outputs)
        return result
    else:
        result = {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }
        result.update(geometry_outputs)
        return result


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
