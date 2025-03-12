import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..af_libs.equivariance import apply_fractional_translation
from ..af_libs.torch_utils.ops.upfirdn2d import upfirdn2d
from ..af_libs.ideal_lpf import UpsampleRFFT, LPF_RFFT
from .flow_utils import flow_warp
from enum import Enum
from typing import Literal, Optional


FILTER_CHOICES = [
    'bilinear',
    'lanczos',
    'ideal',
    'ideal_crop',
    'fourier',
    'fourier_crop',
]
FilterType = Optional[Literal[
    'bilinear',
    'lanczos',
    'ideal',
    'ideal_crop',
    'fourier',
    'fourier_crop'
]]


def gen_valid_mask(shape, ti, tj):
    _, _, h, w = shape
    if ti >= 0:
        ti = int(np.ceil(ti))
        i1, i2 = 0, ti
    else:
        ti = int(np.floor(ti))
        i1, i2 = ti, h

    if tj >= 0:
        tj = int(np.ceil(tj))
        j1, j2 = 0, tj
    else:
        tj = int(np.floor(tj))
        j1, j2 = tj, w
    mask = torch.ones(shape, dtype=torch.float32)
    mask[:, :, i1:i2, :] = 0
    mask[:, :, :, j1:j2] = 0
    return mask


def gen_random_offset(max_offset_i,
                      max_offset_j,
                      int_offset,
                      int_stride,
                      bs=1,
                      min_offset_i=0,
                      min_offset_j=0,
                      ):
    len_i = max_offset_i - min_offset_i
    len_j = max_offset_j - min_offset_j
    if int_offset:
        range_i = int(len_i // int_stride)
        range_j = int(len_j // int_stride)
        img_offset_i = torch.randint(-range_i,
                                     range_i + 1, (bs, )).to(torch.float32)
        img_offset_j = torch.randint(-range_j,
                                     range_j + 1, (bs, )).to(torch.float32)
        img_offset_i *= int_stride
        img_offset_j *= int_stride
    else:
        img_offset_i = (torch.rand((bs, )) * 2 - 1) * len_i
        img_offset_j = (torch.rand((bs, )) * 2 - 1) * len_j
    img_offset_i += min_offset_i
    img_offset_j += min_offset_j
    return img_offset_i, img_offset_j


def image_random_translate(img, img_max_offset_i,
                           img_max_offset_j, int_offset, int_stride):
    n, c, h, w = img.shape
    random_background = torch.rand((n, c, 1, 1)) * 2 - 1
    random_background = random_background.to(
        device=img.device, dtype=img.dtype)
    img_offset_i, img_offset_j = gen_random_offset(
        img_max_offset_i, img_max_offset_j, int_offset, int_stride)

    img_bwd_flow = -torch.tensor(
        [img_offset_i, img_offset_j]).reshape(1, 2, 1, 1).repeat(n, 1, h, w)
    img_bwd_flow = img_bwd_flow.to(img.device)

    warped_img, img_bwd_mask = flow_warp(img, img_bwd_flow, True)

    img_bwd_mask = img_bwd_mask.unsqueeze(1).to(torch.float32)
    warped_img = warped_img * img_bwd_mask + \
        random_background * (1 - img_bwd_mask)

    return warped_img, img_offset_i, img_offset_j, img_bwd_mask


def fourier_shift_batch(image, shift_x, shift_y, device='cuda'):

    # Ensure image is on the correct device
    image = image.to(device)

    # Get the image dimensions
    N, C, H, W = image.shape

    # Fourier transform of the image
    fft_image = torch.fft.fft2(image)

    # Generate frequency grids
    u = torch.fft.fftfreq(W, device=device)
    v = torch.fft.fftfreq(H, device=device)

    # Create frequency grid for each dimension
    U, V = torch.meshgrid(u, v, indexing='ij')

    # Calculate phase shift
    phase_shift = torch.exp(-2j * np.pi * (shift_x * U + shift_y * V))

    # Expand phase shift to match batch and channel dimensions
    phase_shift = phase_shift.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, H, W]

    # Apply the phase shift
    shifted_fft_image = fft_image * phase_shift

    # Inverse Fourier transform to get the shifted image
    shifted_image = torch.fft.ifft2(shifted_fft_image)

    # Return the real part of the result
    return torch.real(shifted_image)


class ImageShifter:
    class BgType(Enum):
        NO_BG = 0
        RANDN = 1
        FULL_COLOR = 2
        ORIGINAL_IMG = 3

    def __init__(self, filter: FilterType = None, upsample_ratio=None):
        '''
        filter choices: None, 'lanczos'
        '''
        if filter is None:
            filter = 'bilinear'
        assert filter in FILTER_CHOICES, f'Wrong filter type {filter}'
        self.__filter = filter
        self.__cache_img = None
        self.__cache_upsampled_img = None
        if filter == 'ideal' or filter == 'ideal_crop':
            assert upsample_ratio is not None
            self.upsample_ratio = upsample_ratio
            self.up_layer = UpsampleRFFT(upsample_ratio)

    def shift(self, img, ti, tj):
        n, _, h, w = img.shape
        if self.__filter == 'lanczos':
            warped_img, mask = apply_fractional_translation(
                img, tj / w, ti / h)
            mask = mask[:, 0:1, :, :]
        elif self.__filter in ['ideal', 'ideal_crop']:

            if self.upsample_ratio == 1:
                self.__cache_upsampled_img = img
            elif self.__cache_img is None or self.__cache_img.data_ptr() != img.data_ptr():
                self.up_layer.to(img.device)
                self.__cache_img = img
                self.__cache_upsampled_img = self.up_layer(img)

            si = int(np.round(ti * self.upsample_ratio))
            sj = int(np.round(tj * self.upsample_ratio))
            warped_img = torch.roll(
                self.__cache_upsampled_img, shifts=(si, sj), dims=(2, 3))

            if self.__filter == 'ideal':
                warped_img = warped_img[:, :,
                                        ::self.upsample_ratio,
                                        ::self.upsample_ratio]
                mask = torch.ones_like(warped_img)
            elif self.__filter == 'ideal_crop':
                upsampled_mask = gen_valid_mask(
                    warped_img.shape, si, sj)
                upsampled_mask = upsampled_mask.to(warped_img.device)
                warped_img *= upsampled_mask
                warped_img = warped_img[:, :,
                                        ::self.upsample_ratio,
                                        ::self.upsample_ratio]
                mask = gen_valid_mask(warped_img.shape, ti, tj)
                mask = mask.to(warped_img.device)
        elif self.__filter == 'fourier':
            warped_img = fourier_shift_batch(img, ti, tj, img.device)
            mask = torch.ones_like(warped_img)
        elif self.__filter == 'fourier_crop':
            warped_img = fourier_shift_batch(img, ti, tj, img.device)
            mask = gen_valid_mask(warped_img.shape, ti, tj)
            mask = mask.to(warped_img.device)
            warped_img *= mask
        else:
            bwd_flow = torch.tensor(
                [-ti, -tj]).reshape(1, 2, 1, 1).repeat(n, 1, h, w)
            bwd_flow = bwd_flow.to(device=img.device)
            warped_img, mask = flow_warp(img, bwd_flow, True)
            mask = mask.unsqueeze(1).to(torch.float32)
        return warped_img, mask

    def translate_with_occ_bg(self, img: torch.Tensor, ti: float, tj: float,
                              bg_type: BgType, mask: torch.Tensor = None,
                              return_mask: bool = False):
        if bg_type == ImageShifter.BgType.RANDN:
            # e.g. Gaussian Noise
            background = torch.randn_like(img)
        elif bg_type == ImageShifter.BgType.FULL_COLOR:
            # Pure color background
            n, c = img.shape[0:2]
            background = torch.rand((n, c, 1, 1)) * 2 - 1
            background = background.to(
                device=img.device, dtype=img.dtype)
        elif bg_type == ImageShifter.BgType.ORIGINAL_IMG:
            background = img
        elif bg_type != ImageShifter.BgType.NO_BG:
            raise ValueError(
                f'No such background type {bg_type} in image shifter')
        warped_img, translate_mask = self.shift(img, ti, tj)
        if mask is None:
            mask = translate_mask
        if bg_type != ImageShifter.BgType.NO_BG:
            warped_img = warped_img * mask + \
                background * (1 - mask)

        if return_mask:
            return warped_img, mask
        else:
            return warped_img

    def image_latent_random_translate(self, img, latent, img_max_offset_i, img_max_offset_j,
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

        warped_img, ti, tj, img_bwd_mask =  \
            image_random_translate(
                img, img_max_offset_i, img_max_offset_j, int_offset, int_stride)

        latent_bwd_mask = F.interpolate(
            img_bwd_mask, scale_factor=1 / downsample_ratio, mode='nearest')
        warped_latent = self.translate_with_occ_bg(
            latent, ti / downsample_ratio, tj / downsample_ratio,
            ImageShifter.FULL_COLOR, latent_bwd_mask)

        return warped_img, warped_latent, img_bwd_mask, latent_bwd_mask


def get_blur_kernel(channels, len=4):
    if len == 4:
        kernel = (1, 3, 3, 1)
    elif len == 5:
        kernel = (1, 3, 6, 3, 1)
    kernel = torch.tensor(kernel, dtype=torch.float32)
    kernel = torch.outer(kernel, kernel)
    kernel /= torch.sum(kernel)
    kernel = kernel.reshape(1, 1, len, len)
    kernel = kernel.repeat(channels, channels, 1, 1)
    return kernel


def upsample_pad_zero(x, scale):
    _, channel, in_h, in_w = x.shape
    x = x.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = x.shape

    x = x.view(-1, in_h, 1, in_w, 1, minor)
    x = F.pad(x, [0, 0, 0, scale - 1, 0, 0, 0, scale - 1])
    x = x.reshape(-1, channel, in_h * scale, in_w * scale)
    return x


class ImageUpsampler():
    def __init__(self, scale=2, mode='nearest', device='cuda'):

        self.scale = scale
        self.mode = mode
        if mode == 'ideal':
            self.up = UpsampleRFFT(scale).to(device)
        elif mode == 'blur':
            self.blur_kernel = get_blur_kernel(1)[0, 0].to(device)
        elif mode == 'learn':
            self.kernel = nn.ConvTranspose2d(1, 1, 4, scale, 1, bias=False)
            self.kernel.weight.data = get_blur_kernel(1, 4) * self.scale ** 2
            self.kernel.to(device)

    def low_pass(self, x):
        if self.mode == 'blur':
            x = upfirdn2d(
                x, self.blur_kernel * 4, 2, padding=(2, 1, 2, 1))
            # x = F.conv2d(x, self.blur_kernel, padding='same')
        elif self.mode == 'ideal':
            x = self.up.recon_filter(x)

        else:
            x = F.interpolate(x, scale_factor=self.scale, mode=self.mode)
        return x

    def upsample(self, x: torch.Tensor):
        # x_input = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        n, c, h, w = x.shape
        x = x.reshape(n * c, 1, h, w)
        if self.mode == 'blur':
            x = upfirdn2d(x, self.blur_kernel * self.scale **
                          2, self.scale, padding=(2, 1, 2, 1))
        elif self.mode == 'ideal':
            x = self.up(x)
        elif self.mode == 'learn':
            # self.kernel.weight.data = self.kernel.weight.data / self.kernel.weight.data.sum() * self.scale**2
            # x = upfirdn2d(x, self.kernel.weight.data[0, 0], self.scale, padding=(2, 1, 2, 1))
            x = self.kernel(x)
        else:
            x = F.interpolate(x, scale_factor=self.scale, mode=self.mode)
        x = x.reshape(n, c, h * self.scale, w * self.scale)
        # mask = torch.zeros_like(x)
        # mask[:, :, ::self.scale, ::self.scale] = 1
        # x = x_input * mask + x * (1 - mask)
        return x


class ImageDownsampler():
    def __init__(self, scale=2, mode='nearest', device='cuda'):

        self.scale = scale
        self.mode = mode
        if mode == 'ideal':
            self.low_pass = LPF_RFFT(scale).to(device)
        elif mode == 'blur':
            self.blur_kernel = get_blur_kernel(1)[0, 0].to(device)

    def downsample(self, x: torch.Tensor):
        # x_input = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        n, c, h, w = x.shape
        x = x.reshape(n * c, 1, h, w)
        if self.mode == 'blur':
            x = upfirdn2d(x, self.blur_kernel, self.scale,
                          padding=(2, 1, 2, 1))
            x = x[:, :, ::2, ::2]
        elif self.mode == 'ideal':
            x = self.low_pass(x)
            x = x[:, :, ::2, ::2]
        else:
            x = F.interpolate(x, scale_factor=1/self.scale, mode=self.mode)
        x = x.reshape(n, c, h // self.scale, w // self.scale)
        return x
