import os
import sys

import numpy as np
from numba import njit
import torch
import torch.nn.functional as F
from einops import rearrange
from .flow_utils_np import (
    flow_warp2, get_intermediate_warp_mask, forward_flow_warp)
from torchvision.utils import save_image
from ..af_libs.equivariance import apply_fractional_translation


from gmflow.gmflow import GMFlow
from gmflow.gm_utils.utils import InputPadder


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def bilinear_sample(img, sample_coords, mode="bilinear", padding_mode="zeros", return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode,
                        padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <=
                                                  1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp(feature, flow, mask=False, mode="bilinear", padding_mode="zeros"):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2
    flow = torch.flip(flow, (1, ))

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]
    grid = grid.to(feature.dtype)
    return bilinear_sample(feature, grid, mode=mode, padding_mode=padding_mode, return_mask=mask)


def flow_warp_with_occ_bg(img, flow, mask, is_randn, filter=None):
    '''
    filter choices: [None, 'lanczos']
    '''

    if is_randn:
        # e.g. Gaussian Noise
        background = torch.randn_like(img)
    else:
        # Pure color background
        n, c = img.shape[0:2]
        background = torch.rand((n, c, 1, 1)) * 2 - 1
        background = background.to(
            device=img.device, dtype=img.dtype)
    if filter == 'lanczos':
        _, _, h, w = img.shape
        tx = -flow[0, 1, 0, 0].item() / w
        ty = -flow[0, 0, 0, 0].item() / h
        warped_img, _ = apply_fractional_translation(
            img, tx, ty)
    else:
        warped_img = flow_warp(img, flow)
    warped_img = warped_img * mask + \
        background * (1 - mask)
    return warped_img


def flow_revserse_map(feature, flow):
    # Flow: nchw -> nhwc
    flow = torch.flip(flow, (1, ))
    flow = rearrange(flow, 'n c h w -> n h w c')

    n, c, h, w = feature.shape
    i, j = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]
    pos = torch.stack([i, j], dim=2).unsqueeze(0)
    pos = pos.to(device=flow.device)
    pos2 = torch.round(pos + flow).to(torch.int64)
    pos2 = pos2[:, :, :, 0].clamp(
        0, h - 1) * w + pos2[:, :, :, 1].clamp(0, w - 1)
    pos2 = pos2.repeat(n*c, 1, 1).flatten(1)
    feature_flat = feature.reshape(n * c, h * w)
    return torch.gather(feature_flat, 1, pos2).reshape(n, c, h, w)


def forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.01, beta=0.5):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow
    # (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + \
        torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ.unsqueeze(1), bwd_occ.unsqueeze(1)


@torch.no_grad()
def get_warped_and_mask(flow_model, image1, image2, image3=None, pixel_consistency=False):
    if image3 is None:
        image3 = image1
    padder = InputPadder(image1.shape, padding_factor=16)
    image1, image2 = padder.pad(image1.cuda(), image2.cuda())
    results_dict = flow_model(
        image1, image2, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=True
    )
    flow_pr = results_dict["flow_preds"][-1]  # [B, 2, H, W]
    fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)  # [1, 2, H, W]
    bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)  # [1, 2, H, W]
    fwd_occ, bwd_occ = forward_backward_consistency_check(
        fwd_flow, bwd_flow, beta=1)  # [1, H, W] float
    if pixel_consistency:
        warped_image1 = flow_warp(image1, bwd_flow)
        bwd_occ = torch.clamp(
            bwd_occ + (abs(image2 - warped_image1).mean(dim=1)
                       > 255 * 0.25).float(), 0, 1
        ).unsqueeze(0)
    bwd_flow = torch.flip(bwd_flow, (1, ))
    warped_results = flow_warp(image3, bwd_flow)
    return warped_results, bwd_occ, bwd_flow


@torch.no_grad()
def predict_flow(flow_model, image1, image2):
    padder = InputPadder(image1.shape, padding_factor=8)
    image1, image2 = padder.pad(image1.cuda(), image2.cuda())
    results_dict = flow_model(
        image1, image2, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=True
    )
    flow_pr = results_dict["flow_preds"][-1]  # [B, 2, H, W]
    fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)  # [1, 2, H, W]
    bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)  # [1, 2, H, W]
    fwd_occ, bwd_occ = forward_backward_consistency_check(
        fwd_flow, bwd_flow)  # [1, H, W] float
    return fwd_flow, fwd_occ, bwd_flow, bwd_occ


@torch.no_grad()
def alpha_warp(flow_model, image1, image2, alpha):
    fwd_flow, fwd_occ, bwd_flow, bwd_occ = predict_flow(
        flow_model, image1, image2)
    warped_results = flow_warp2(image1, fwd_flow * alpha, fwd_occ)
    return warped_results, fwd_flow, fwd_occ, bwd_flow, bwd_occ


def upsample_noise(noise, ratio):
    n, c, h, w = noise.shape
    z = torch.randn(n, c, ratio * h, ratio *
                    w).to(dtype=noise.dtype, device=noise.device)
    z_mean = z.unfold(2, ratio, ratio).unfold(3, ratio, ratio).mean((4, 5))
    z_mean = F.interpolate(z_mean, scale_factor=ratio, mode='nearest')
    x = F.interpolate(noise, scale_factor=ratio, mode='nearest')
    return x / ratio + z - z_mean


def collect_noise_pixel(noise, bwd_occ, sidelength):
    sl = sidelength
    n, c, h, w = noise.shape
    res = torch.randn_like(noise) * bwd_occ + noise * (1 - bwd_occ)
    res = res.reshape(n, c, h // sl, sl, w // sl, sl)
    res = rearrange(res, 'n c h a w b -> n c h w a b')
    res = torch.sum(res, dim=(-1, -2)) / sidelength
    return res


def continuous_noise_warp(high_res_noise, fwd_flow, fwd_occ, alpha, noise_ratio=8):
    # fwd_occ = torch.zeros_like(fwd_occ)
    bwd_flow, bwd_occ = get_intermediate_warp_mask(fwd_flow, fwd_occ, alpha)
    # bwd_occ = torch.zeros_like(bwd_occ)
    warpped_noise = flow_warp(high_res_noise, bwd_flow)
    noise = collect_noise_pixel(warpped_noise, bwd_occ, noise_ratio)

    return noise


def continuous_noise_warp_bwd(high_res_noise, bwd_flow,     bwd_occ, noise_ratio=8, flow_ratio=1):
    bwd_flow = F.interpolate(bwd_flow, scale_factor=flow_ratio)
    bwd_occ = F.interpolate(bwd_occ, scale_factor=flow_ratio, mode='nearest')
    warpped_noise = flow_warp(high_res_noise, bwd_flow)
    noise = collect_noise_pixel(warpped_noise, bwd_occ, noise_ratio)
    return noise


def get_patch_moving_flow(img_template, region_box, displacement, alpha=1):
    n, _, h, w = img_template.shape
    u, d, l, r = region_box
    di, dj = displacement
    bwd_flow = torch.zeros(n, 2, h, w).to(
        device=img_template.device, dtype=torch.float32)
    bwd_occ = torch.zeros(n, 1, h, w).to(
        device=img_template.device, dtype=torch.float32)
    bwd_occ[:, :, u:d, l:r] = 1
    u = int(np.round(u + di * alpha))
    d = int(np.round(d + di * alpha))
    l = int(np.round(l + dj * alpha))
    r = int(np.round(r + dj * alpha))

    bwd_flow[:, 0, u:d, l:r] = -di * alpha
    bwd_flow[:, 1, u:d, l:r] = -dj * alpha
    bwd_occ[:, :, u:d, l:r] = 0
    return bwd_flow, bwd_occ


def continuous_noise_fwd_warp(high_res_noise, fwd_flow, alpha, noise_ratio=8):
    fwd_flow = fwd_flow * alpha
    warpped_noise, bwd_occ = forward_flow_warp(high_res_noise, fwd_flow)
    noise = collect_noise_pixel(warpped_noise, bwd_occ, noise_ratio)

    return noise


def __image_random_translate(img, img_max_offset_i,
                             img_max_offset_j, int_offset, int_stride):
    n, c, h, w = img.shape
    random_background = torch.rand((n, c, 1, 1)) * 2 - 1
    random_background = random_background.to(
        device=img.device, dtype=img.dtype)

    if int_offset:
        range_i = int(img_max_offset_i // int_stride)
        range_j = int(img_max_offset_j // int_stride)
        img_offset_i = torch.randint(-range_i,
                                     range_i + 1, (1, )).to(torch.float32)
        img_offset_j = torch.randint(-range_j,
                                     range_j + 1, (1, )).to(torch.float32)
        img_offset_i *= int_stride
        img_offset_j *= int_stride
    else:
        img_offset_i = (torch.rand((1, )) * 2 - 1) * img_max_offset_i
        img_offset_j = (torch.rand((1, )) * 2 - 1) * img_max_offset_j
    img_bwd_flow = -torch.tensor(
        [img_offset_i, img_offset_j]).reshape(1, 2, 1, 1).repeat(n, 1, h, w)
    img_bwd_flow = img_bwd_flow.to(img.device)

    warped_img, img_bwd_mask = flow_warp(img, img_bwd_flow, True)
    # warped_img, img_bwd_mask = apply_fractional_translation(
    #     img, img_offset_j.item(), img_offset_i.item())

    img_bwd_mask = img_bwd_mask.unsqueeze(1).to(torch.float32)
    warped_img = warped_img * img_bwd_mask + \
        random_background * (1 - img_bwd_mask)

    return warped_img, img_bwd_flow, img_bwd_mask


def image_random_translate(img, img_max_offset_i, img_max_offset_j,
                           batch_size=1, int_offset=False, int_stride=1):

    img = img.repeat(batch_size, 1, 1, 1)
    warped_img, _, _, =  \
        __image_random_translate(
            img, img_max_offset_i, img_max_offset_j, int_offset, int_stride)
    return warped_img


def image_latent_random_translate(img, latent, img_max_offset_i, img_max_offset_j,
                                  batch_size=1, int_offset=False, align_latent=False):
    n, c, h, w = img.shape
    n2, c2, h2, w2 = latent.shape
    assert n == n2
    assert h * w2 == w * h2
    downsample_ratio = h / h2
    assert downsample_ratio == np.round(downsample_ratio)

    img = img.repeat(batch_size, 1, 1, 1)
    latent = latent.repeat(batch_size, 1, 1, 1)
    n *= batch_size

    if align_latent:
        int_stride = downsample_ratio
    else:
        int_stride = 1

    warped_img, img_bwd_flow, img_bwd_mask =  \
        __image_random_translate(
            img, img_max_offset_i, img_max_offset_j, int_offset, int_stride)

    latent_bwd_flow = img_bwd_flow / downsample_ratio
    latent_bwd_flow = F.interpolate(
        latent_bwd_flow, scale_factor=1 / downsample_ratio, mode='bilinear')
    latent_bwd_mask = F.interpolate(
        img_bwd_mask, scale_factor=1 / downsample_ratio, mode='nearest')
    warped_latent = flow_warp_with_occ_bg(
        latent, latent_bwd_flow, latent_bwd_mask, True, 'lanczos')

    return warped_img, warped_latent, latent_bwd_mask


def noise_image_random_translate(img, noise, img_max_offset_i, img_max_offset_j,
                                 noise_upsample=True, batch_size=1,
                                 int_offset=False):
    n, c, h, w = img.shape
    n2, c2, h2, w2 = noise.shape
    assert n == n2
    assert h * w2 == w * h2
    img_noise_ratio = h / h2
    assert img_noise_ratio == np.round(img_noise_ratio)
    img_noise_ratio = int(np.round(img_noise_ratio))
    img = img.repeat(batch_size, 1, 1, 1)
    noise = noise.repeat(batch_size, 1, 1, 1)
    n *= batch_size
    warped_img, img_bwd_flow, img_bwd_mask =  \
        __image_random_translate(
            img, img_max_offset_i, img_max_offset_j, int_offset)

    if noise_upsample:
        high_res_noise = upsample_noise(noise, img_noise_ratio)
        warped_noise = continuous_noise_warp_bwd(
            high_res_noise, img_bwd_flow, 1 - img_bwd_mask, img_noise_ratio)

    else:
        noise_bwd_flow = img_bwd_flow / img_noise_ratio
        noise_bwd_flow = F.interpolate(
            noise_bwd_flow, scale_factor=1 / img_noise_ratio, mode='bilinear')
        noise_bwd_mask = F.interpolate(
            img_bwd_mask, scale_factor=1 / img_noise_ratio, mode='nearest')
        warped_noise = flow_warp_with_occ_bg(
            noise, noise_bwd_flow, noise_bwd_mask, True)

    return warped_img, warped_noise


def forward_upsample_flow_warp(img, fwd_flow, scale=8):
    from afldm.af_libs.ideal_lpf import UpsampleRFFT
    upsample = UpsampleRFFT(scale)
    img = upsample(img)
    warped_img, occ = forward_flow_warp(img, fwd_flow)
    warped_img = warped_img[:, :, ::scale, ::scale]
    occ = occ[:, :, ::scale, ::scale]
    return warped_img, occ
