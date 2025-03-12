import numpy as np
import torch
from numba import njit


@njit
def clip_int(x: int, low: int, high: int) -> int:
    if x < low:
        x = low
    if x > high:
        x = high
    return x


@njit
def _flow_warp(img: np.ndarray, bwd_flow: np.ndarray) -> np.ndarray:
    n, c, h, w = img.shape
    res = np.zeros_like(img)
    for ni in range(n):
        for ci in range(c):
            for i in range(h):
                for j in range(w):
                    prev_i = int(np.round(i + bwd_flow[ni, 0, i, j]))
                    prev_j = int(np.round(j + bwd_flow[ni, 1, i, j]))
                    prev_i = clip_int(prev_i, 0, h - 1)
                    prev_j = clip_int(prev_j, 0, w - 1)
                    res[ni, ci, i, j] = img[ni, ci, prev_i, prev_j]
    return res


def flow_warp(img, bwd_flow):
    img_ = img.cpu().numpy()
    bwd_flow_ = bwd_flow.cpu().numpy()
    img2 = _flow_warp(img_, bwd_flow_)
    img2 = torch.from_numpy(img2).to(dtype=img.dtype, device=img.device)
    return img2


@njit
def _flow_warp2(img: np.ndarray, fwd_flow: np.ndarray) -> np.ndarray:
    n, c, h, w = img.shape
    res = np.zeros_like(img)
    for ni in range(n):
        for ci in range(c):
            for i in range(h):
                for j in range(w):
                    crt_i = int(np.round(i + fwd_flow[ni, 0, i, j]))
                    crt_j = int(np.round(j + fwd_flow[ni, 1, i, j]))
                    crt_i = clip_int(crt_i, 0, h - 1)
                    crt_j = clip_int(crt_j, 0, w - 1)
                    res[ni, ci, crt_i, crt_j] += img[ni, ci, i, j]
    return res


def flow_warp2(img, fwd_flow, fwd_occ):
    # if img.shape[1:] != fwd_flow.shape[1:]:
    img = img * (1 - fwd_occ)
    img_ = img.cpu().numpy()
    fwd_flow_ = fwd_flow.cpu().numpy()
    img2 = _flow_warp2(img_, fwd_flow_)
    img2 = torch.from_numpy(img2).to(dtype=img.dtype, device=img.device)
    return img2


@njit
def _get_intermediate_warp_mask(fwd_flow, fwd_occ):
    n, _, h, w = fwd_flow.shape
    cnt_matrix = np.zeros((n, 1, h, w), dtype=np.int32)

    bwd_occ = np.ones_like(fwd_occ)
    bwd_flow = np.zeros_like(fwd_flow)
    for ni in range(n):
        for i in range(h):
            for j in range(w):
                crt_i = int(np.round(i + fwd_flow[ni, 0, i, j]))
                crt_j = int(np.round(j + fwd_flow[ni, 1, i, j]))
                crt_i = clip_int(crt_i, 0, h - 1)
                crt_j = clip_int(crt_j, 0, w - 1)
                if not fwd_occ[ni, 0, i, j]:
                    cnt_matrix[ni, 0, crt_i, crt_j] += 1
                    bwd_flow[ni, 0, crt_i, crt_j] = -fwd_flow[ni, 0, i, j]
                    bwd_flow[ni, 1, crt_i, crt_j] = -fwd_flow[ni, 1, i, j]

    for ni in range(n):
        for i in range(h):
            for j in range(w):
                if cnt_matrix[ni, 0, i, j] == 1:
                    bwd_occ[ni, 0, i, j] = 0

    return bwd_flow, bwd_occ


def get_intermediate_warp_mask(fwd_flow, fwd_occ, alpha):

    fwd_flow = fwd_flow * alpha
    fwd_flow_np = fwd_flow.cpu().numpy()
    fwd_occ_np = fwd_occ.cpu().numpy()
    bwd_flow, bwd_occ = _get_intermediate_warp_mask(fwd_flow_np, fwd_occ_np)
    bwd_flow = torch.from_numpy(bwd_flow).to(
        dtype=fwd_flow.dtype, device=fwd_flow.device)
    bwd_occ = torch.from_numpy(bwd_occ).to(
        dtype=fwd_occ.dtype, device=fwd_occ.device)
    return bwd_flow, bwd_occ


@njit
def _update_bilinear_coef(c, h, w, v, x_out, cnt, i, j, grid_i, grid_j):
    if grid_i >= 0 and grid_i < h and grid_j >= 0 and grid_j < w:
        coef = (1 - abs(i - grid_i)) * (1 - abs(j - grid_j))
        cnt[grid_i][grid_j] += coef
        for ci in range(c):
            x_out[ci][grid_i][grid_j] += v[ci] * coef


@njit
def _forward_flow_warp(x, fwd_flow):
    n, c, h, w = x.shape
    cnt_matrix = np.zeros((n, h, w), dtype=x.dtype)
    bwd_occ = np.ones((n, 1, h, w), dtype=x.dtype)

    res = np.zeros_like(x)
    for ni in range(n):
        for i in range(h):
            for j in range(w):
                crt_i = i + fwd_flow[ni, 0, i, j]
                crt_j = j + fwd_flow[ni, 1, i, j]
                i1 = int(crt_i)
                i2 = i1 + 1
                j1 = int(crt_j)
                j2 = j1 + 1

                v = x[ni, :, i, j]
                x_out = res[ni]
                crt_cnt = cnt_matrix[ni]
                _update_bilinear_coef(
                    c, h, w, v, x_out, crt_cnt, crt_i, crt_j, i1, j1)
                _update_bilinear_coef(
                    c, h, w, v, x_out, crt_cnt, crt_i, crt_j, i2, j1)
                _update_bilinear_coef(
                    c, h, w, v, x_out, crt_cnt, crt_i, crt_j, i1, j2)
                _update_bilinear_coef(
                    c, h, w, v, x_out, crt_cnt, crt_i, crt_j, i2, j2)

    for ni in range(n):
        for i in range(h):
            for j in range(w):
                if cnt_matrix[ni][i][j] > 0:
                    bwd_occ[ni, 0, i, j] = 0
                    # for ci in range(c):
                    #     res[ni][ci][i][j] /= cnt_matrix[ni][i][j]

    return res, bwd_occ


def forward_flow_warp(img, fwd_flow):
    img_ = img.cpu().numpy()
    fwd_flow_ = fwd_flow.cpu().numpy()
    img2, bwd_occ = _forward_flow_warp(img_, fwd_flow_)
    img2 = torch.from_numpy(img2).to(dtype=img.dtype, device=img.device)
    bwd_occ = torch.from_numpy(bwd_occ).to(dtype=img.dtype, device=img.device)
    return img2, bwd_occ
