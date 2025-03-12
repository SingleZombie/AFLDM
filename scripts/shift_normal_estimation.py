import argparse
import inspect

import torch
from tqdm import tqdm

from afldm.af_modules.af_api import (make_af_controlnet, make_af_unet,
                                     make_af_vae_from_config)
from afldm.io_utils import image_to_tensor, save_gif_from_tensors
from afldm.pipelines.cross_frame_attn import (AttnState,
                                              CrossFrameAttnProcessor,
                                              get_unet_attn_processors,
                                              set_unet_attn_processor)
from afldm.pipelines.normal_control_pipeline import NormControlPipeline
from afldm.shift_utils.shifters import ImageShifter


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--shift_steps', type=int, default=16)
    parser.add_argument('--output_path', type=str,
                        default='results/shift_nomal.gif')
    parser.add_argument('--input_path', type=str,
                        default='assets/normal_input_1.png')
    return parser.parse_args()


def vae_encode(vae, x):
    x = vae.encode(x).latent_dist.sample()
    x = x * vae.config.scaling_factor
    return x


def vae_decode(vae, x):
    x = vae.decode(x / vae.config.scaling_factor, return_dict=False)[0]
    return x


@torch.no_grad()
def shift_normal_controlnet(pipeline,
                            num_shift_steps=16,
                            output_path='results/shift_normal.gif',
                            input_path='assets/normal_input_1.png'):
    device = pipeline.device

    vae = pipeline.vae
    unet = pipeline.unet
    controlnet = pipeline.controlnet
    scheduler = pipeline.scheduler
    pipeline.set_progress_bar_config(disable=True)

    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        '',
        'cuda',
        1,
        False,
        ''
    )

    # It usually equals 8
    downsample_ratio = 2**(len(vae.up_block_types) - 1)

    latent_shifter = ImageShifter('ideal_crop', downsample_ratio)
    image_shifter = ImageShifter()

    attn_state = AttnState()
    attn_processor_dict = {}

    # For recovery
    prev_dict = {}

    processors = get_unet_attn_processors(unet)
    for k in processors:
        prev_dict[k] = processors[k]
        attn_processor_dict[k] = CrossFrameAttnProcessor(
            attn_state)
    set_unet_attn_processor(unet, attn_processor_dict)

    eta = 0
    accepts_eta = "eta" in set(inspect.signature(
        scheduler.step).parameters.keys())
    extra_kwargs = {}
    if accepts_eta:
        extra_kwargs["eta"] = eta

    def denoise(latents, image):
        # Predefined fixed timestep.
        t = 999
        latent_model_input = latents
        latent_model_input = scheduler.scale_model_input(
            latent_model_input, t)

        attn_state.set_timestep(t)

        control_model_input = latent_model_input
        controlnet_prompt_embeds = prompt_embeds

        down_block_res_samples, mid_block_res_sample = controlnet(
            control_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=image,
            conditioning_scale=1.0,
            guess_mode=False,
            return_dict=False,
        )

        # predict the noise residual
        output = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        return output

    # Input
    tensor = image_to_tensor(input_path, None).to(device)

    # Get low resolution image
    cond_latent = vae_encode(vae, tensor)
    latent = torch.randn_like(cond_latent)

    attn_state.reset()
    denoised_latent = denoise(latent, cond_latent)
    attn_state.to_load()

    rec_init_img = vae_decode(vae, denoised_latent)

    res_imgs = []

    offsets = torch.linspace(1/downsample_ratio,
                             num_shift_steps/downsample_ratio, num_shift_steps)
    iterator = tqdm(range(offsets.shape[0]))

    for i in iterator:
        tj = offsets[i]
        ti = 0

        shifted_cond_latent, mask = latent_shifter.shift(
            cond_latent, ti, tj)
        shifted_latent, _ = latent_shifter.shift(latent, ti, tj)

        denoised_shifted_latent = denoise(shifted_latent, shifted_cond_latent)
        shifted_rec_img_gt, _ = image_shifter.shift(
            rec_init_img, ti * downsample_ratio, tj * downsample_ratio)

        # Visualization
        img_input = vae_decode(vae, shifted_cond_latent * mask)
        img_2 = vae_decode(vae, denoised_shifted_latent * mask)
        img_ori = shifted_rec_img_gt
        img_diff = torch.abs((img_2 - img_ori))
        img_cat = torch.cat(
            (img_input, img_2, img_ori, img_diff), -2)
        res_imgs.append(img_cat.cpu())

    # Recover the original processors
    processors = get_unet_attn_processors(unet)
    for k in processors:
        attn_processor_dict[k] = prev_dict[k]
    set_unet_attn_processor(unet, attn_processor_dict)

    save_gif_from_tensors(res_imgs, output_path, denorm=True)


if __name__ == '__main__':
    args = parse_args()

    pipe = NormControlPipeline.from_pretrained(
        'SingleZombie/alias_free_norm_controlnet', trust_remote_code=True
    ).to('cuda')
    pipe.set_progress_bar_config(disable=True)

    make_af_unet(pipe.unet)
    make_af_vae_from_config(pipe.vae)
    make_af_controlnet(pipe.controlnet)

    shift_normal_controlnet(pipe, args.shift_steps,
                            args.output_path, args.input_path)
