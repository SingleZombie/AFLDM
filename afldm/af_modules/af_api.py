from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from diffusers.models.downsampling import Downsample2D
from diffusers.models.upsampling import Upsample2D

from .af_blocks import (AliasFreeDownsample2D, AliasFreeUpsample2D,
                        WarpedNonlinearity)


def wrap_nonlinearity(nonlinearity):
    return WarpedNonlinearity(nonlinearity)


def replace_upsampler(ori_upsampler: Upsample2D):
    block = AliasFreeUpsample2D(ori_upsampler.channels, ori_upsampler.use_conv,
                                ori_conv=ori_upsampler.conv,
                                out_channels=ori_upsampler.out_channels)
    return block


def replace_downsampler(ori_downsampler: Downsample2D):
    block = AliasFreeDownsample2D(ori_downsampler.channels, ori_downsampler.use_conv,
                                  out_channels=ori_downsampler.out_channels,
                                  padding=ori_downsampler.padding,
                                  ori_conv=ori_downsampler.conv)
    return block


def wrap_resblock_nonlinearity(block):
    for i in range(len(block.resnets)):
        block.resnets[i].nonlinearity = wrap_nonlinearity(
            block.resnets[i].nonlinearity)


def make_af_vae(vae: AutoencoderKL,
                mod_mid_act=True,
                mod_down_filtered_act=[True, True, True, True],
                mod_up_filtered_act=[True, True, True, True],
                mod_resampling_layer=[True, True, True]):
    encoder = vae.encoder
    decoder = vae.decoder

    mod_downsampling_list = list(reversed(mod_resampling_layer))
    for i, block in enumerate(encoder.down_blocks):
        if block.downsamplers is not None and mod_downsampling_list[i]:
            block.downsamplers[0] = replace_downsampler(
                block.downsamplers[0])
        if mod_down_filtered_act[i]:
            wrap_resblock_nonlinearity(block)

    if mod_mid_act:
        wrap_resblock_nonlinearity(encoder.mid_block)

    if mod_mid_act:
        wrap_resblock_nonlinearity(decoder.mid_block)

    for i, block in enumerate(decoder.up_blocks):
        if mod_up_filtered_act[i]:
            wrap_resblock_nonlinearity(block)
        if block.upsamplers is not None and mod_resampling_layer[i]:
            block.upsamplers[0] = replace_upsampler(block.upsamplers[0])


def make_af_vae_from_config(vae: AutoencoderKL):
    make_af_vae(vae, mod_mid_act=vae.config.mid_act,
                mod_down_filtered_act=vae.config.down_filtered_act,
                mod_up_filtered_act=vae.config.up_filtered_act,
                mod_resampling_layer=vae.config.up_rescale)


def make_af_unet(unet: UNet2DConditionModel):
    for i, block in enumerate(unet.down_blocks):
        if block.downsamplers is not None:
            block.downsamplers[0] = replace_downsampler(
                block.downsamplers[0])
        wrap_resblock_nonlinearity(block)

    wrap_resblock_nonlinearity(unet.mid_block)

    for i, block in enumerate(unet.up_blocks):
        if block.upsamplers is not None:
            block.upsamplers[0] = replace_upsampler(
                block.upsamplers[0])
        wrap_resblock_nonlinearity(block)


def make_af_controlnet(model: ControlNetModel):
    for i, block in enumerate(model.down_blocks):
        if block.downsamplers is not None:
            block.downsamplers[0] = replace_downsampler(
                block.downsamplers[0])
        wrap_resblock_nonlinearity(block)

    wrap_resblock_nonlinearity(model.mid_block)
