from typing import Tuple
from diffusers import AutoencoderKL, ModelMixin
from diffusers.configuration_utils import register_to_config

from ..af_modules.af_api import make_af_vae


class AliasFreeAutoencoderKL(AutoencoderKL, ModelMixin):

    @register_to_config
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 down_block_types: Tuple[str] = ...,
                 up_block_types: Tuple[str] = ...,
                 block_out_channels: Tuple[int] = ...,
                 layers_per_block: int = 1,
                 act_fn: str = "silu",
                 latent_channels: int = 4,
                 norm_num_groups: int = 32,
                 sample_size: int = 32,
                 scaling_factor: float = 0.18215,
                 shift_factor: float = None,
                 latents_mean: Tuple[float] = None,
                 latents_std: Tuple[float] = None,
                 force_upcast: float = True,
                 use_quant_conv: bool = True,
                 use_post_quant_conv: bool = True,
                 mid_block_add_attention: bool = True,
                 mod_mid_act=True,
                down_filtered_act=[True, True, True, True],
                up_filtered_act=[True, True, True, True],
                up_rescale=[True, True, True]):
        super().__init__(in_channels, out_channels, down_block_types,
                         up_block_types, block_out_channels, layers_per_block,
                         act_fn, latent_channels, norm_num_groups,
                         sample_size, scaling_factor, shift_factor,
                         latents_mean, latents_std, force_upcast,
                         use_quant_conv, use_post_quant_conv,
                         mid_block_add_attention)
        make_af_vae(self, mod_mid_act, down_filtered_act,
                    up_filtered_act, up_rescale)

    @property
    def downsample_ratio(self):
        return 2 ** (len(self.config.block_out_channels) - 1)

    def encode_scale(self, x):
        x = self.encode(x).latent_dist.sample()
        x = x * self.config.scaling_factor
        return x

    def decode_scale(self, x):
        x = self.decode(x / self.config.scaling_factor).sample
        return x
