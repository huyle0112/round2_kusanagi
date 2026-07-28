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

import os
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torch.nn.functional as F
import torchvision
import csv
import json
import wandb
import time
from os import makedirs
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf

import lpipsPyTorch as lpips
import random
from random import randint
from utils.loss_utils import edge_aware_loss, l1_loss, ssim
from utils.gaussianpro import GaussianProAnchorBuilder
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# torch.set_num_threads(32)
lpips_fn = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    global lpips_fn
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    gaussianpro = None
    gaussianpro_initial_anchors = int(gaussians.get_anchor.shape[0])
    gaussianpro_max_anchors = int(
        gaussianpro_initial_anchors * opt.gaussianpro_max_anchor_multiplier
    )
    if opt.use_gaussianpro:
        if opt.gaussianpro_voxel_factor <= 0:
            raise ValueError("gaussianpro_voxel_factor must be positive")
        if opt.gaussianpro_min_consistent_views > opt.gaussianpro_neighbors:
            raise ValueError(
                "gaussianpro_min_consistent_views cannot exceed "
                "gaussianpro_neighbors"
            )
        if not (
            0.0 <= opt.gaussianpro_near_quantile
            < opt.gaussianpro_far_quantile
            <= 1.0
        ):
            raise ValueError(
                "GaussianPro depth quantiles must satisfy "
                "0 <= near < far <= 1"
            )
        if (
            opt.gaussianpro_relaxed_min_views
            > opt.gaussianpro_min_consistent_views
        ):
            raise ValueError(
                "gaussianpro_relaxed_min_views cannot exceed "
                "gaussianpro_min_consistent_views"
            )
        if (
            opt.gaussianpro_relaxed_min_views < 1
            or opt.gaussianpro_relaxed_min_views
            > opt.gaussianpro_neighbors
        ):
            raise ValueError(
                "gaussianpro_relaxed_min_views must be between 1 and "
                "gaussianpro_neighbors"
            )
        if opt.gaussianpro_edge_radius <= 0:
            raise ValueError("gaussianpro_edge_radius must be positive")
        if opt.gaussianpro_scaffold_fallback_interval < 1:
            raise ValueError(
                "gaussianpro_scaffold_fallback_interval must be positive"
            )
        if not (
            opt.gaussianpro_start_iter
            < opt.gaussianpro_add_until_iter
            <= opt.gaussianpro_refine_until_iter
            < opt.iterations
        ):
            raise ValueError(
                "GaussianPro schedule must satisfy start < add_until <= "
                "refine_until < iterations"
            )
        gaussianpro = GaussianProAnchorBuilder(
            scene.getTrainCameras(),
            gaussians.get_anchor.detach(),
            num_neighbors=opt.gaussianpro_neighbors,
            graph_samples=opt.gaussianpro_graph_samples,
            min_overlap=opt.gaussianpro_min_overlap,
            downsample=opt.gaussianpro_downsample,
            patch_radius=opt.gaussianpro_patch_radius,
            patchmatch_iterations=opt.gaussianpro_patchmatch_iterations,
            opacity_threshold=opt.gaussianpro_opacity_threshold,
            coverage_threshold=opt.gaussianpro_coverage_threshold,
            min_consistent_views=opt.gaussianpro_min_consistent_views,
            adaptive_views=opt.gaussianpro_adaptive_views,
            relaxed_min_views=opt.gaussianpro_relaxed_min_views,
            near_quantile=opt.gaussianpro_near_quantile,
            far_quantile=opt.gaussianpro_far_quantile,
            edge_radius=opt.gaussianpro_edge_radius,
            max_photo_error=opt.gaussianpro_max_photo_error,
            reprojection_threshold=opt.gaussianpro_reprojection_threshold,
            depth_consistency_threshold=(
                opt.gaussianpro_depth_consistency_threshold
            ),
            normal_consistency_threshold=(
                opt.gaussianpro_normal_consistency_threshold
            ),
            depth_discrepancy_threshold=(
                opt.gaussianpro_depth_discrepancy_threshold
            ),
            edge_residual_priority=(
                opt.gaussianpro_edge_residual_priority
            ),
            max_anchors_per_step=opt.gaussianpro_max_anchors_per_step,
            min_proposals_per_step=1,
            use_plane_ncc=True,
            propagate_source_views=True,
            seed=opt.gaussianpro_seed,
        )
        graph_sizes = [
            len(neighbors)
            for neighbors in gaussianpro.graph.values()
        ]
        graph_mean = (
            sum(graph_sizes) / len(graph_sizes) if graph_sizes else 0.0
        )
        message = (
            "GaussianPro anchor growth enabled: "
            f"{len(graph_sizes)} reference cameras, "
            f"{graph_mean:.1f} neighbours/reference, "
            f"add {opt.gaussianpro_start_iter}-"
            f"{opt.gaussianpro_add_until_iter}, refine until "
            f"{opt.gaussianpro_refine_until_iter}, "
            f"interval {opt.gaussianpro_interval}, "
            f"downsample {opt.gaussianpro_downsample}, "
            f"views {opt.gaussianpro_min_consistent_views}->"
            f"{opt.gaussianpro_relaxed_min_views} at depth tails/edge, "
            f"Scaffold fallback "
            f"{opt.gaussianpro_scaffold_fallback_interval} iters, "
            f"anchor budget {gaussianpro_max_anchors}"
        )
        if logger:
            logger.info(message)
        else:
            print(message)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        # network gui not available in scaffold-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        gaussianpro_result = None
        gaussianpro_lifecycle = None
        gaussianpro_step = (
            gaussianpro is not None
            and iteration >= opt.gaussianpro_start_iter
            and iteration < opt.gaussianpro_refine_until_iter
            and iteration % max(1, opt.gaussianpro_interval) == 0
        )
        if gaussianpro_step:
            gaussianpro_allow_add = (
                iteration < opt.gaussianpro_add_until_iter
            )
            reference_camera = gaussianpro.next_reference()
            gaussianpro_result, proposed_points, _proposed_normals = (
                gaussianpro.run(
                    reference_camera,
                    gaussians,
                    pipe,
                    background,
                    render,
                    prefilter_voxel,
                    require_new_anchor=gaussianpro_allow_add,
                )
            )
            gaussianpro_lifecycle = gaussians.manage_gaussianpro_anchors(
                proposed_points,
                gaussianpro.last_point_confidence,
                iteration=iteration,
                allow_add=gaussianpro_allow_add,
                max_total_anchors=gaussianpro_max_anchors,
                voxel_size=(
                    gaussians.voxel_size
                    * opt.gaussianpro_voxel_factor
                ),
                refine_radius=(
                    gaussians.voxel_size
                    * opt.gaussianpro_refine_radius_factor
                ),
                refine_rate=opt.gaussianpro_refine_rate,
                refine_scaffold_anchors=(
                    opt.gaussianpro_refine_scaffold_anchors
                ),
                confidence_decay=opt.gaussianpro_confidence_decay,
                prune=(
                    iteration % max(
                        1, opt.gaussianpro_prune_interval
                    )
                    == 0
                ),
                prune_confidence=opt.gaussianpro_prune_confidence,
                prune_opacity=opt.gaussianpro_prune_opacity,
                prune_grace_iterations=(
                    opt.gaussianpro_prune_grace_iters
                ),
            )
            gaussianpro_result.added_count = gaussianpro_lifecycle["added"]
            if logger:
                logger.info(
                    "[ITER %d] Propagation %s: candidates=%d, "
                    "photo=%d, consistent=%d, proposed=%d, "
                    "added=%d, refined=%d, scaffold_supported=%d, "
                    "pruned=%d, gp=%d, scaffold=%d, total=%d, "
                    "photo_error=%.4f, views=%.2f",
                    iteration,
                    gaussianpro_result.reference_name,
                    gaussianpro_result.candidate_count,
                    gaussianpro_result.photometric_count,
                    gaussianpro_result.consistent_count,
                    gaussianpro_result.proposed_count,
                    gaussianpro_result.added_count,
                    gaussianpro_lifecycle["refined"],
                    gaussianpro_lifecycle["scaffold_supported"],
                    gaussianpro_lifecycle["pruned"],
                    gaussianpro_lifecycle["gaussianpro_total"],
                    gaussianpro_lifecycle["scaffold_total"],
                    gaussianpro_lifecycle["total"],
                    gaussianpro_result.mean_photo_error,
                    gaussianpro_result.mean_consistent_views,
                )
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        gaussianpro_active = (
            gaussianpro is not None
            and iteration >= opt.gaussianpro_start_iter
            and iteration < opt.gaussianpro_refine_until_iter
        )
        gaussianpro_plane_step = (
            gaussianpro_active
            and hasattr(viewpoint_cam, "gaussianpro_normal_target")
        )

        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            visible_mask=voxel_visible_mask,
            retain_grad=retain_grad,
            return_normal=gaussianpro_plane_step,
            geometry_downsample=opt.gaussianpro_downsample,
        )
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg
        edge_loss_value = None
        edge_lambda = 0.0
        if (
            iteration >= opt.edge_loss_start_iter
            and (
                opt.lambda_edge_init > 0
                or opt.lambda_edge_final > 0
            )
        ):
            edge_progress = (
                (iteration - opt.edge_loss_start_iter)
                / max(1, opt.iterations - opt.edge_loss_start_iter)
            )
            edge_progress = min(1.0, max(0.0, edge_progress))
            edge_lambda = (
                opt.lambda_edge_init
                + edge_progress
                * (opt.lambda_edge_final - opt.lambda_edge_init)
            )
            edge_loss_value = edge_aware_loss(
                image,
                gt_image,
                top_weight=opt.edge_top_weight,
                border_weight=opt.edge_border_weight,
                blur_floor=opt.edge_blur_floor,
                block_size=opt.edge_block_size,
            )
            loss = loss + edge_lambda * edge_loss_value

        gaussianpro_flatness_ratio = None
        gaussianpro_normal_l1 = None
        gaussianpro_normal_cos = None
        gaussianpro_feature_l1 = None
        gaussianpro_feature_cos = None
        if gaussianpro_active:
            # Keep GP shape regularization off Scaffold fallback anchors.
            gp_gaussian_mask = None
            if (
                gaussians._gaussianpro_anchor_mask.numel()
                == gaussians.get_anchor.shape[0]
            ):
                visible_gp = gaussians._gaussianpro_anchor_mask[
                    voxel_visible_mask
                ]
                visible_gp_offsets = visible_gp.unsqueeze(1).expand(
                    -1, gaussians.n_offsets
                ).reshape(-1)
                gp_gaussian_mask = visible_gp_offsets[
                    offset_selection_mask
                ]
            if gp_gaussian_mask is not None and gp_gaussian_mask.any():
                # Penalize scale ratio, not raw scale, to avoid covariance
                # collapse.
                sorted_scaling = scaling[gp_gaussian_mask].sort(
                    dim=1
                ).values
                gaussianpro_flatness_ratio = (
                    sorted_scaling[:, 0]
                    / sorted_scaling[:, 1].clamp_min(1e-8)
                ).mean()
                flatness_penalty = (
                    F.relu(
                        gaussianpro_flatness_ratio
                        - opt.gaussianpro_flatness_target
                    )
                    + F.relu(
                        opt.gaussianpro_flatness_floor
                        - gaussianpro_flatness_ratio
                    )
                )
                loss = (
                    loss
                    + opt.lambda_gaussianpro_flatness
                    * flatness_penalty
                )
            if iteration % max(1, opt.gaussianpro_feature_interval) == 0:
                (
                    gaussianpro_feature_l1,
                    gaussianpro_feature_cos,
                ) = gaussians.gaussianpro_feature_loss(
                    sample_count=opt.gaussianpro_feature_samples
                )
                loss = (
                    loss
                    + opt.lambda_gaussianpro_feature_l1
                    * gaussianpro_feature_l1
                    + opt.lambda_gaussianpro_feature_cos
                    * gaussianpro_feature_cos
                )

        if gaussianpro_plane_step:
            predicted_normal = F.normalize(
                render_pkg["render_normal"], dim=0, eps=1e-6
            )
            target_normal = (
                viewpoint_cam.gaussianpro_normal_target.to(
                    device=predicted_normal.device,
                    dtype=predicted_normal.dtype,
                )
            )
            target_mask = viewpoint_cam.gaussianpro_target_mask.to(
                device=predicted_normal.device
            )
            if target_normal.shape[-2:] != predicted_normal.shape[-2:]:
                target_normal = F.interpolate(
                    target_normal.unsqueeze(0),
                    size=predicted_normal.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                target_mask = (
                    F.interpolate(
                        target_mask.float()[None, None],
                        size=predicted_normal.shape[-2:],
                        mode="nearest",
                    )[0, 0]
                    > 0.5
                )
            target_normal = F.normalize(
                target_normal, dim=0, eps=1e-6
            )
            if target_mask.any():
                normal_difference = (
                    predicted_normal - target_normal
                ).abs().sum(dim=0)
                angular_difference = 1.0 - (
                    predicted_normal * target_normal
                ).sum(dim=0).clamp(-1.0, 1.0)
                gaussianpro_normal_l1 = normal_difference[
                    target_mask
                ].mean()
                gaussianpro_normal_cos = angular_difference[
                    target_mask
                ].mean()
                loss = (
                    loss
                    + opt.lambda_gaussianpro_normal_l1
                    * gaussianpro_normal_l1
                    + opt.lambda_gaussianpro_normal_cos
                    * gaussianpro_normal_cos
                )

        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if gaussianpro_flatness_ratio is not None:
                    postfix["GP-ratio"] = f"{gaussianpro_flatness_ratio.item():.3f}"
                if edge_loss_value is not None:
                    postfix["Edge"] = f"{edge_loss_value.item():.4f}"
                if gaussianpro_normal_l1 is not None:
                    postfix["GPF-l1"] = (
                        f"{gaussianpro_normal_l1.item():.4f}"
                    )
                    postfix["GPF-cos"] = (
                        f"{gaussianpro_normal_cos.item():.4f}"
                    )
                if gaussianpro_feature_l1 is not None:
                    postfix["GP-feat"] = (
                        f"{gaussianpro_feature_l1.item():.4f}"
                    )
                if gaussianpro_result is not None:
                    postfix["GP-add"] = gaussianpro_result.added_count
                    postfix["GP-ref"] = gaussianpro_lifecycle["refined"]
                    postfix["GP-del"] = gaussianpro_lifecycle["pruned"]
                    postfix["GP-cons"] = (
                        gaussianpro_result.consistent_count
                    )
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
                if tb_writer and gaussianpro_flatness_ratio is not None:
                    tb_writer.add_scalar(
                        f"{dataset_name}/gaussianpro/flatness_ratio",
                        gaussianpro_flatness_ratio.item(),
                        iteration,
                    )
                if tb_writer and gaussianpro_normal_l1 is not None:
                    tb_writer.add_scalar(
                        f"{dataset_name}/gaussianpro/normal_l1",
                        gaussianpro_normal_l1.item(),
                        iteration,
                    )
                    tb_writer.add_scalar(
                        f"{dataset_name}/gaussianpro/normal_cos",
                        gaussianpro_normal_cos.item(),
                        iteration,
                    )
                if tb_writer and gaussianpro_feature_l1 is not None:
                    tb_writer.add_scalar(
                        f"{dataset_name}/gaussianpro/feature_l1",
                        gaussianpro_feature_l1.item(),
                        iteration,
                    )
                    tb_writer.add_scalar(
                        f"{dataset_name}/gaussianpro/feature_cos",
                        gaussianpro_feature_cos.item(),
                        iteration,
                    )
                if tb_writer and gaussianpro_result is not None:
                    tb_writer.add_scalar(
                        f"{dataset_name}/gaussianpro/proposed",
                        gaussianpro_result.proposed_count,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        f"{dataset_name}/gaussianpro/added",
                        gaussianpro_result.added_count,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        f"{dataset_name}/gaussianpro/refined",
                        gaussianpro_lifecycle["refined"],
                        iteration,
                    )
                    tb_writer.add_scalar(
                        f"{dataset_name}/gaussianpro/pruned",
                        gaussianpro_lifecycle["pruned"],
                        iteration,
                    )
                    tb_writer.add_scalar(
                        f"{dataset_name}/gaussianpro/photo_error",
                        gaussianpro_result.mean_photo_error,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        f"{dataset_name}/gaussianpro/consistent_views",
                        gaussianpro_result.mean_consistent_views,
                        iteration,
                    )
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                dataset_name,
                iteration,
                Ll1,
                loss,
                l1_loss,
                opt.lambda_dssim,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                wandb,
                logger,
                validation_sample_count=getattr(dataset, "validation_sample_count", 0),
                validation_seed=getattr(dataset, "validation_seed", 42),
            )
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    scaffold_fallback_step = (
                        opt.use_gaussianpro
                        and opt.gaussianpro_scaffold_fallback
                        and iteration
                        % max(
                            1,
                            opt.gaussianpro_scaffold_fallback_interval,
                        )
                        == 0
                    )
                    if not opt.use_gaussianpro or scaffold_fallback_step:
                        scaffold_added = gaussians.adjust_anchor(
                            check_interval=(
                                opt.gaussianpro_scaffold_fallback_interval
                                if opt.use_gaussianpro
                                else opt.update_interval
                            ),
                            success_threshold=opt.success_threshold,
                            grad_threshold=opt.densify_grad_threshold,
                            # GaussianPro owns pruning while hybrid training is
                            # active, so fallback only grows image-driven
                            # anchors and cannot bypass its grace period.
                            min_opacity=(
                                -1.0
                                if opt.use_gaussianpro
                                else opt.min_opacity
                            ),
                            max_total_anchors=(
                                gaussianpro_max_anchors
                                if opt.use_gaussianpro
                                else None
                            ),
                        )
                        if (
                            opt.use_gaussianpro
                            and scaffold_added > 0
                            and logger
                        ):
                            logger.info(
                                "[ITER %d] Scaffold fallback added=%d, "
                                "total=%d/%d",
                                iteration,
                                scaffold_added,
                                int(gaussians.get_anchor.shape[0]),
                                gaussianpro_max_anchors,
                            )
            cleanup_iteration = (
                opt.gaussianpro_refine_until_iter
                if gaussianpro is not None
                else opt.update_until
            )
            if iteration == cleanup_iteration:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def _append_monitor_row(path, fieldnames, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def training_report(
    tb_writer,
    dataset_name,
    iteration,
    Ll1,
    loss,
    l1_loss,
    lambda_dssim,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    wandb=None,
    logger=None,
    validation_sample_count=0,
    validation_seed=42,
):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/pixel_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    if iteration == 1 or iteration % 100 == 0:
        _append_monitor_row(
            Path(scene.model_path) / "train_curve.csv",
            ["iteration", "train_l1", "train_total_loss", "anchors"],
            {
                "iteration": iteration,
                "train_l1": float(Ll1.item()),
                "train_total_loss": float(loss.item()),
                "anchors": int(scene.gaussians.get_anchor.shape[0]),
            },
        )

    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    if iteration in testing_iterations:
        global lpips_fn
        if lpips_fn is None:
            lpips_fn = lpips.LPIPS("vgg").to("cuda")
            lpips_fn.eval()
            lpips_fn.requires_grad_(False)

        scene.gaussians.eval()
        torch.cuda.empty_cache()
        train_cameras = scene.getTrainCameras()
        val_cameras = scene.getTestCameras()
        if not val_cameras and validation_sample_count > 0:
            # Proxy validation: all cameras remain available for optimization.
            # We only select fixed cameras for repeatable metric monitoring.
            sample_count = min(validation_sample_count, len(train_cameras))
            rng = random.Random(validation_seed)
            val_cameras = rng.sample(train_cameras, sample_count)
            val_ids = {id(camera) for camera in val_cameras}
            remaining = [
                camera for camera in train_cameras
                if id(camera) not in val_ids
            ]
            train_pool = remaining if len(remaining) >= sample_count else train_cameras
            train_eval_cameras = rng.sample(
                train_pool, min(sample_count, len(train_pool))
            )
        else:
            train_sample_size = min(
                len(train_cameras),
                max(1, len(val_cameras)),
            )
            train_eval_cameras = random.Random(validation_seed).sample(
                train_cameras, train_sample_size
            )
        validation_configs = (
            {"name": "train_eval", "cameras": train_eval_cameras},
            {"name": "val", "cameras": val_cameras},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 3):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_fn(
                        image.unsqueeze(0), gt_image.unsqueeze(0)
                    ).mean().double()

                camera_count = len(config["cameras"])
                psnr_test /= camera_count
                l1_test /= camera_count
                ssim_test /= camera_count
                lpips_test /= camera_count
                photo_loss = (
                    (1.0 - lambda_dssim) * l1_test
                    + lambda_dssim * (1.0 - ssim_test)
                )
                metrics_row = {
                    "iteration": iteration,
                    "split": config["name"],
                    "count": camera_count,
                    "photo_loss": float(photo_loss.item()),
                    "l1": float(l1_test.item()),
                    "psnr": float(psnr_test.item()),
                    "ssim": float(ssim_test.item()),
                    "lpips": float(lpips_test.item()),
                    "anchors": int(scene.gaussians.get_anchor.shape[0]),
                }
                _append_monitor_row(
                    Path(scene.model_path) / "validation_metrics.csv",
                    [
                        "iteration",
                        "split",
                        "count",
                        "photo_loss",
                        "l1",
                        "psnr",
                        "ssim",
                        "lpips",
                        "anchors",
                    ],
                    metrics_row,
                )
                logger.info(
                    "\n[ITER %d] %s (%d views): loss=%.6f "
                    "PSNR=%.4f SSIM=%.5f LPIPS=%.5f",
                    iteration,
                    config["name"],
                    camera_count,
                    metrics_row["photo_loss"],
                    metrics_row["psnr"],
                    metrics_row["ssim"],
                    metrics_row["lpips"],
                )
                if tb_writer:
                    for metric_name in (
                        "photo_loss", "l1", "psnr", "ssim", "lpips"
                    ):
                        tb_writer.add_scalar(
                            f"{dataset_name}/{config['name']}/{metric_name}",
                            metrics_row[metric_name],
                            iteration,
                        )
                if wandb is not None:
                    wandb.log(
                        {
                            f"{config['name']}_{key}": value
                            for key, value in metrics_row.items()
                            if isinstance(value, (int, float))
                        }
                    )

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()

        # Determine filename: use original filename with extension if present (from CSV)
        out_name = view.image_name if '.' in view.image_name else '{0:05d}.png'.format(idx)
        name_list.append(out_name)
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, out_name))
        torchvision.utils.save_image(errormap, os.path.join(error_path, out_name))
        torchvision.utils.save_image(gt, os.path.join(gts_path, out_name))
        per_view_dict[out_name] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    global lpips_fn
    # Ensure LPIPS model exists for evaluation metrics
    if lpips_fn is None:
        lpips_fn = lpips.LPIPS('vgg').to('cuda')
        lpips_fn.eval()
        lpips_fn.requires_grad_(False)

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    

    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info("\nTraining complete.")

    logger.info("Run render.py and metrics.py separately to render and evaluate the trained model.")
