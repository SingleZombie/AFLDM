import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .fir_resblock import wrap_nonlinearity, LPF_RFFT, KaiserDownsample


class DownsampleLPFConv(nn.Conv2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 use_kaiser=False) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device, dtype)
        self.stride = 1

        self.use_kaiser = use_kaiser

        if use_kaiser:
            self.down_layer = KaiserDownsample(2)
        else:
            self.lpf = LPF_RFFT()

    def forward(self, x):
        x = super().forward(x)
        if self.use_kaiser:
            x = self.down_layer(x)
        else:
            x = self.lpf(x)
            x = x[:, :, ::2, ::2]
        return x


class Discriminator(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_channels=3, hidden_channels=512, depth=6,
                 use_bn=False, antialias=False, use_kaiser=False, mod_act=True):
        super().__init__()

        use_bias = not use_bn
        norm_cls = nn.InstanceNorm2d if not use_bn else nn.BatchNorm2d

        nonlinearity = nn.LeakyReLU(0.2, True)
        if antialias:
            isFFT = not use_kaiser
            if mod_act:
                nonlinearity = wrap_nonlinearity(
                    nonlinearity, isFFT=isFFT, use_kaiser=use_kaiser)

        downsample_conv_cls = nn.Conv2d
        ex_kwargs = {}
        if antialias:
            downsample_conv_cls = DownsampleLPFConv
            ex_kwargs = {'use_kaiser': use_kaiser}

        d = max(depth - 3, 3)
        layers = [
            downsample_conv_cls(in_channels, hidden_channels // (2**d),
                                kernel_size=4, stride=2, padding=1,
                                **ex_kwargs),
            nonlinearity,
        ]
        for i in range(depth - 1):
            c_in = hidden_channels // (2 ** max((d - i), 0))
            c_out = hidden_channels // (2 ** max((d - 1 - i), 0))

            layers.append(
                downsample_conv_cls(c_in, c_out, kernel_size=4, stride=2,
                                    padding=1, bias=use_bias, **ex_kwargs))
            layers.append(norm_cls(c_out))
            layers.append(nonlinearity)
        c_in = c_out
        c_out = hidden_channels
        layers.append(
            nn.Conv2d(
                c_in, c_out, kernel_size=4, stride=1, padding=1, bias=use_bias
            )
        )
        layers.append(norm_cls(c_out))
        layers.append(nonlinearity)
        # layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Conv2d(
            c_out, 1, kernel_size=4, stride=1, padding=1
        ))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        return x
