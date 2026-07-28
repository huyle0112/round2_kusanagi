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
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def edge_aware_loss(
    network_output,
    gt,
    top_weight=1.15,
    border_weight=1.10,
    top_fraction=0.20,
    border_fraction=0.15,
    blur_floor=0.35,
    block_size=64,
):
    """Weighted first/second-order loss with conservative blur reliability.

    Spatial emphasis is applied only to gradients. Low-gradient blocks receive
    less weight, preventing a locally blurred observation from teaching a soft
    edge while retaining its RGB/SSIM supervision and camera geometry.
    """
    pred_gray = network_output.mean(dim=0, keepdim=True)
    gt_gray = gt.mean(dim=0, keepdim=True)
    height, width = gt_gray.shape[-2:]

    gt_dx = gt_gray[..., :, 1:] - gt_gray[..., :, :-1]
    gt_dy = gt_gray[..., 1:, :] - gt_gray[..., :-1, :]
    pred_dx = pred_gray[..., :, 1:] - pred_gray[..., :, :-1]
    pred_dy = pred_gray[..., 1:, :] - pred_gray[..., :-1, :]

    with torch.no_grad():
        magnitude = torch.zeros_like(gt_gray)
        magnitude[..., :, 1:] += gt_dx.abs()
        magnitude[..., 1:, :] += gt_dy.abs()
        kernel = max(1, min(int(block_size), height, width))
        local_energy = F.avg_pool2d(
            magnitude.unsqueeze(0),
            kernel_size=kernel,
            stride=kernel,
            ceil_mode=True,
        )
        positive = local_energy[local_energy > 0]
        threshold = (
            torch.quantile(positive, 0.20)
            if positive.numel()
            else local_energy.new_tensor(0.0)
        )
        reliability = torch.where(
            local_energy <= threshold,
            local_energy.new_tensor(float(blur_floor)),
            local_energy.new_tensor(1.0),
        )
        reliability = F.interpolate(
            reliability,
            size=(height, width),
            mode="nearest",
        ).squeeze(0)

        spatial = torch.ones_like(gt_gray)
        top_rows = max(1, int(round(height * float(top_fraction))))
        border_x = max(1, int(round(width * float(border_fraction))))
        border_y = max(1, int(round(height * float(border_fraction))))
        spatial[..., :top_rows, :] *= float(top_weight)
        spatial[..., :, :border_x] *= float(border_weight)
        spatial[..., :, width - border_x :] *= float(border_weight)
        spatial[..., :border_y, :] *= float(border_weight)
        spatial[..., height - border_y :, :] *= float(border_weight)
        weight = spatial * reliability

    first_order = (
        ((pred_dx - gt_dx).abs() * weight[..., :, 1:]).mean()
        + ((pred_dy - gt_dy).abs() * weight[..., 1:, :]).mean()
    )
    laplacian_kernel = gt_gray.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    pred_laplacian = F.conv2d(
        pred_gray.unsqueeze(0), laplacian_kernel, padding=1
    ).squeeze(0)
    gt_laplacian = F.conv2d(
        gt_gray.unsqueeze(0), laplacian_kernel, padding=1
    ).squeeze(0)
    second_order = (
        (pred_laplacian - gt_laplacian).abs() * weight
    ).mean()
    return first_order + 0.5 * second_order

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

