from typing import Tuple
from .af_vae import AliasFreeAutoencoderKL


class AutoencoderKLTraining(AliasFreeAutoencoderKL):
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
                 mod_rescale=False,
                 fft_rescale=False,
                 fft_filtered_act=True,
                 mid_act=False,
                 down_filtered_act=[False, False, False, False],
                 up_filtered_act=[False, False, False, False],
                 down_cutoff_list=[1.0, 1.0, 1.0, 1.0],
                 up_cutoff_list=[1.0, 1.0, 1.0, 1.0],
                 up_rescale=[True, True, True],
                 use_kaiser=False,
                 fix=False):
        super().__init__(in_channels, out_channels, down_block_types,
                         up_block_types, block_out_channels, layers_per_block,
                         act_fn, latent_channels, norm_num_groups,
                         sample_size, scaling_factor, shift_factor,
                         latents_mean, latents_std, force_upcast,
                         use_quant_conv, use_post_quant_conv,
                         mid_block_add_attention, mod_rescale,
                         fft_rescale, fft_filtered_act, mid_act,
                         down_filtered_act, up_filtered_act, down_cutoff_list,
                         up_cutoff_list, up_rescale, use_kaiser, fix)

    def forward(self, x, is_encoding=True):
        if is_encoding:
            return self.encode(x)
        else:
            return self.decode(x)
