import argparse
import inspect

import torch
from tqdm import tqdm

from afldm.af_modules.af_api import make_af_unet, make_af_vae_from_config
from afldm.io_utils import image_to_tensor, save_gif_from_tensors
from afldm.pipelines.cross_frame_attn import (AttnState,
                                              CrossFrameAttnProcessor,
                                              get_unet_attn_processors,
                                              set_unet_attn_processor)
from afldm.pipelines.i2sb_pipeline import I2SBLDMPipeline
from afldm.shift_utils.shifters import ImageShifter
from afldm.af_libs.superresolution import build_sr4x


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--shift_steps', type=int, default=16)
    parser.add_argument('--output_path', type=str,
                        default='results/shift_ldm_sr.gif')
    parser.add_argument('--input_path', type=str,
                        default='assets/bear_hr.jpg')
    return parser.parse_args()


def vae_encode(vae, x):
    x = vae.encode(x).latent_dist.mode()
    x = x * vae.config.scaling_factor
    return x


def vae_decode(vae, x):
    x = vae.decode(x / vae.config.scaling_factor, return_dict=False)[0]
    return x


@torch.no_grad()
def shift_ldm_sr(pipeline, num_inference_steps=50, num_shift_steps=16, output_path='results/shift_ldm.gif', input_path=None):
    device = pipeline.device

    sr_func = build_sr4x(device, 'bicubic', 256)

    vae = pipeline.vae
    unet = pipeline.unet
    scheduler = pipeline.scheduler
    pipeline.set_progress_bar_config(disable=True)

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

    def denoise(latents):
        latents = latents.to(device)
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        for i, t in enumerate(timesteps):
            if i == num_inference_steps - 1:
                break

            # expand the latents if we are doing classifier free guidance
            latent_model_input = latents
            latent_model_input = scheduler.scale_model_input(
                latent_model_input, t)

            attn_state.set_timestep(t)

            noise_pred = unet(
                latent_model_input,
                t,
                return_dict=False
            )[0]

            latents = scheduler.step(
                noise_pred, t, latents,
                is_ode=True, generator=None).prev_sample

        return latents

    # Input
    tensor = image_to_tensor(input_path, None).to(device)

    # Get low resolution image
    tensor = sr_func(tensor).clip(-1, 1)
    init_latent = vae_encode(vae, tensor)
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)

    attn_state.reset()
    denoised_latent = denoise(init_latent)
    attn_state.to_load()
    rec_init_img = vae_decode(vae, denoised_latent)

    res_imgs = []

    offsets = torch.linspace(1/downsample_ratio,
                             num_shift_steps/downsample_ratio, num_shift_steps)
    iterator = tqdm(range(offsets.shape[0]))

    for i in iterator:
        tj = offsets[i]
        ti = 0

        shifted_init_latent, mask = latent_shifter.shift(
            init_latent, ti, tj)

        denoised_shifted_latent = denoise(shifted_init_latent)
        shifted_rec_img_gt, _ = image_shifter.shift(
            rec_init_img, ti * downsample_ratio, tj * downsample_ratio)

        # Visualization
        img_input = vae_decode(vae, shifted_init_latent * mask)
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

    pipe = I2SBLDMPipeline.from_pretrained(
        'SingleZombie/alias_free_ldm_sr', trust_remote_code=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)

    make_af_unet(pipe.unet)
    make_af_vae_from_config(pipe.vae)

    shift_ldm_sr(pipe, args.num_inference_steps, args.shift_steps,
                 args.output_path, args.input_path)
