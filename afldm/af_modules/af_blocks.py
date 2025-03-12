import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.downsampling import Downsample2D
from diffusers.models.upsampling import Upsample2D

from ..af_libs.ideal_lpf import LPF_RFFT, UpsampleRFFT


class WarpedNonlinearity(nn.Module):
    def __init__(self, nonlinearity):
        super().__init__()
        self.up_layer = UpsampleRFFT()
        self.lpf = LPF_RFFT(1/2)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        if x.ndim < 4:
            return self.nonlinearity(x)

        x = self.up_layer(x)
        x = self.nonlinearity(x)
        x = self.lpf(x)
        x = x[:, :, ::2, ::2]

        return x


class WarpedConvIn(nn.Conv2d):
    def __init__(self, conv: nn.Conv2d, wraped_nonlinearity):
        super().__init__(conv.in_channels, conv.out_channels,
                         conv.kernel_size, conv.stride, conv.padding)
        self.weight = conv.weight
        self.bias = conv.bias
        self.nonlinearity = wraped_nonlinearity

    def forward(self, x):
        x = super().forward(x)
        x = self.nonlinearity(x)
        return x


class AliasFreeUpsample2D(Upsample2D):
    def __init__(self, channels: int,
                 use_conv: bool = False,
                 use_conv_transpose: bool = False,
                 out_channels: int = None,
                 name: str = "conv",
                 kernel_size: int = None,
                 padding=1,
                 norm_type=None,
                 eps=None,
                 elementwise_affine=None,
                 bias=True,
                 interpolate=True,
                 ori_conv=None):
        super().__init__(channels, use_conv, use_conv_transpose, out_channels, name,
                         kernel_size, padding, norm_type, eps, elementwise_affine, bias, interpolate)
        self.up_layer = UpsampleRFFT()
        self.conv = ori_conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size=None, *args, **kwargs
    ) -> torch.FloatTensor:

        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(
                hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            hidden_states = self.up_layer(hidden_states)

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states


class AliasFreeDownsample2D(Downsample2D):

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels=None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        ori_conv=None
    ):

        super().__init__(channels, use_conv, out_channels, padding, name,
                         kernel_size, norm_type, eps, elementwise_affine, bias)
        self.conv = ori_conv
        self.conv.stride = 1
        self.lpf = LPF_RFFT()

        if self.Conv2d_0 is not None:
            self.Conv2d_0 = None

    def forward(self, hidden_states: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(
                hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv and self.padding == 0:
            pad = (1, 1, 1, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)
        hidden_states = self.lpf(hidden_states)
        hidden_states = hidden_states[:, :, ::2, ::2]

        return hidden_states
