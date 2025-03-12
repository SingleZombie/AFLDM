import argparse
import inspect

import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from afldm.af_modules.af_api import make_af_unet, make_af_vae_from_config
from afldm.io_utils import image_to_tensor, save_gif_from_tensors
from afldm.pipelines.cross_frame_attn import (AttnState,
                                              CrossFrameAttnProcessor,
                                              get_unet_attn_processors,
                                              set_unet_attn_processor)
from afldm.pipelines.ldm_pipeline import MyLDMPipeline
from afldm.shift_utils.shifters import ImageShifter


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--shift_steps', type=int, default=16)
    parser.add_argument('--output_path', type=str,
                        default='results/shift_ldm.gif')
    parser.add_argument('--input_path', type=str, default=None)
    return parser.parse_args()


'''
example command:
python scripts/shift_ldm_ffhq.py \
    --num_inference_steps 50 \
    --shift_steps 16 \
    --output_path results/shift_ldm.gif
'''


def vae_encode(vae, x):
    x = vae.encode(x).latent_dist.sample()
    x = x * vae.config.scaling_factor
    return x


def vae_decode(vae, x):
    x = vae.decode(x / vae.config.scaling_factor, return_dict=False)[0]
    return x


@torch.no_grad()
def shift_ldm(pipeline, num_inference_steps=50, num_shift_steps=16, output_path='results/shift_ldm.gif', input_path=None):
    device = pipeline.device
    generator = None

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
                noise_pred, t, latents, return_dict=False,
                **extra_kwargs)[0]

        return latents

    if input_path is not None:
        tensor = image_to_tensor(
            input_path, (unet.config.sample_size, unet.config.sample_size))
        tensor = vae_encode(vae, tensor)

        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        init_latent = pipeline.ddim_inversion(tensor, bar=False)
    else:
        init_latent = randn_tensor(
            (1, unet.config.in_channels,
             unet.config.sample_size, unet.config.sample_size),
            device=device,
            generator=generator)

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
        img_2 = vae_decode(vae, denoised_shifted_latent * mask)
        img_ori = shifted_rec_img_gt
        img_diff = torch.abs((img_2 - img_ori))
        img_cat = torch.cat((img_2, img_ori, img_diff), -2)
        res_imgs.append(img_cat.cpu())

    # Recover the original processors
    processors = get_unet_attn_processors(unet)
    for k in processors:
        attn_processor_dict[k] = prev_dict[k]
    set_unet_attn_processor(unet, attn_processor_dict)

    save_gif_from_tensors(res_imgs, output_path, denorm=True)


if __name__ == '__main__':
    args = parse_args()

    pipe = MyLDMPipeline.from_pretrained(
        'SingleZombie/alias_free_ldm_ffhq').to('cuda')
    pipe.set_progress_bar_config(disable=True)

    make_af_unet(pipe.unet)
    make_af_vae_from_config(pipe.vae)

    shift_ldm(pipe, args.num_inference_steps, args.shift_steps,
              args.output_path, args.input_path)
