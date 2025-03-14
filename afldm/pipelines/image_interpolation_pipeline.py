# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from types import MethodType

import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
from ..shift_utils.flow_utils import (predict_flow, upsample_noise,
                                      continuous_noise_fwd_warp, forward_flow_warp)
from ..af_libs.ideal_lpf import UpsampleRFFT
from gmflow.gmflow.gmflow import GMFlow

from .cross_frame_attn import AttnState, CrossFrameAttnProcessor
from torchvision.utils import save_image

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + \
        (1 - guidance_rescale) * noise_cfg
    return noise_cfg


@torch.no_grad()
def slerp(p0, p1, fract_mixing: float):
    r""" Copied from lunarring/latentblending
    Helper function to correctly mix two random variables using spherical interpolation.
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0: 
            First tensor for interpolation
        p1: 
            Second tensor for interpolation
        fract_mixing: float 
            Mixing coefficient of interval [0, 1]. 
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """
    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'

    p0 = p0.double()
    p1 = p1.double()

    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1+epsilon, 1-epsilon)

    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0*s0 + p1*s1

    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()

    return interp


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError(
            "Could not access latents of provided encoder_output")


class ImageInterpolationPipeline(StableDiffusionPipeline):

    def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler: KarrasDiffusionSchedulers, safety_checker: StableDiffusionSafetyChecker, feature_extractor: CLIPImageProcessor, image_encoder: CLIPVisionModelWithProjection = None, requires_safety_checker=True):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker,
                         feature_extractor, image_encoder, requires_safety_checker)
        self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
        self.safety_checker = None
        self.flow_model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type="swin",
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        ).to("cuda")
        checkpoint = torch.load(
            "gmflow/gmflow_sintel-0c07dcb3.pth", map_location=lambda storage, loc: storage)
        weights = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.flow_model.load_state_dict(weights, strict=False)
        self.flow_model.eval()

    @torch.no_grad()
    def ddim_invert_step(self, latent, cond, scale, i):
        do_classifier_free_guidance = True if scale != 1.0 else False
        timesteps = reversed(self.scheduler.timesteps)
        ori_dtype = latent.dtype
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            cond_batch = cond.repeat(latent.shape[0], 1, 1)
            alpha_prod_t = self.scheduler.alphas_cumprod[timesteps[i]]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else self.scheduler.final_alpha_cumprod
            )

            mu = alpha_prod_t ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            input_latent = torch.cat(
                [latent] * 2) if do_classifier_free_guidance else latent

            eps_list = []
            for j in range(input_latent.shape[0]):
                eps = self.unet(
                    input_latent[j:j+1], timesteps[i], encoder_hidden_states=cond_batch[j:j+1]).sample
                eps_list.append(eps)
            eps = torch.cat(eps_list)

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = eps.chunk(2)
                eps = noise_pred_uncond + scale * \
                    (noise_pred_text - noise_pred_uncond)

            pred_x0 = (latent - sigma_prev * eps) / mu_prev
            latent = mu * pred_x0 + sigma * eps
        latent = latent.to(ori_dtype)
        return latent

    @torch.no_grad()
    def ddim_inversion(self, latent, cond, scale, bar=True):
        timesteps = reversed(self.scheduler.timesteps)
        do_classifier_free_guidance = True if scale != 1.0 else False
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            if bar:
                iterator = tqdm(timesteps, desc="DDIM inversion")
            else:
                iterator = timesteps
            for i, t in enumerate(iterator):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                input_latent = torch.cat(
                    [latent] * 2) if do_classifier_free_guidance else latent
                eps = self.unet(
                    input_latent, t, encoder_hidden_states=cond_batch).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = eps.chunk(2)
                    eps = noise_pred_uncond + scale * \
                        (noise_pred_text - noise_pred_uncond)

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps

        return latent

    @torch.no_grad()
    def image2latent(self, image):
        device = self.vae.device
        dtype = self.vae.dtype
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0)
        image = image.to(device=device, dtype=dtype)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def __call__(
        self,
        image1,
        image2,
        num_frames,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[
            int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        warp_method: int = 0,
        enable_interp: bool = False,
        inv_prompt='',
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of IP-adapters.
                Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should contain the negative image embedding
                if `do_classifier_free_guidance` is set to `True`.
                If not provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        guidance_scale = 1.0

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        image1_pt = self.image_processor.preprocess(
            image1, height=height, width=width).to(device)

        image2_pt = self.image_processor.preprocess(
            image2, height=height, width=width).to(device)

        f_flow, f_occ, b_flow, b_occ = predict_flow(
            self.flow_model, image1_pt, image2_pt)

        # [W, H] -> [H, W]
        f_flow = torch.flip(f_flow, (1, ))
        b_flow = torch.flip(b_flow, (1, ))

        ds_scale = 8
        f_flow_ds = F.interpolate(
            f_flow / ds_scale, scale_factor=1/ds_scale, mode='nearest')
        f_occ_ds = F.interpolate(
            f_occ, scale_factor=1/ds_scale, mode='nearest')
        b_flow_ds = F.interpolate(
            b_flow / ds_scale, scale_factor=1/ds_scale, mode='nearest')
        b_occ_ds = F.interpolate(
            b_occ, scale_factor=1/ds_scale, mode='nearest')

        self.attn_state = AttnState()
        attn_processor_dict = {}
        for k in self.unet.attn_processors.keys():
            attn_processor_dict[k] = CrossFrameAttnProcessor(
                self.attn_state, enable_interp)
        self.unet.set_attn_processor(attn_processor_dict)

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get(
                "scale", None) if self.cross_attention_kwargs is not None else None
        )
        # self._do_classifier_free_guidance = True
        # self._guidance_scale = 7.5

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        prompt_embeds_inv, negative_prompt_embeds_inv = self.encode_prompt(
            inv_prompt,
            device,
            num_images_per_prompt,
            False,
            '',
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        alphas = torch.linspace(0, 1, num_frames)

        image_latent_1 = self.image2latent(image1_pt)
        image_latent_2 = self.image2latent(image2_pt)

        latents = self.prepare_latents(
            num_frames,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        latents[0] = self.ddim_inversion(
            image_latent_1, prompt_embeds_inv, 1)[0]
        latents[-1] = self.ddim_inversion(
            image_latent_2, prompt_embeds_inv, 1)[0]

        if warp_method != 1:
            upsampler = UpsampleRFFT(ds_scale)
            high_res_noise_0 = upsampler(latents[0].unsqueeze(0))
            high_res_noise_1 = upsampler(latents[-1].unsqueeze(0))
        else:
            high_res_noise_0 = upsample_noise(latents[0].unsqueeze(0), 8)
            high_res_noise_1 = upsample_noise(latents[-1].unsqueeze(0), 8)

        occ_bg1 = torch.randn_like(high_res_noise_0)
        occ_bg2 = occ_bg1

        for i in range(1, num_frames - 1):
            alpha = alphas[i]

            if warp_method == 0:
                warped_1, tmp_occ = forward_flow_warp(
                    high_res_noise_0, f_flow * alpha)

                warped_1 = warped_1 * (1 - tmp_occ) + tmp_occ * occ_bg1
                warped_1 = warped_1[:, :, ::ds_scale, ::ds_scale]

                warped_2, tmp_occ = forward_flow_warp(
                    high_res_noise_1, b_flow * (1-alpha))
                warped_2 = warped_2 * (1 - tmp_occ) + tmp_occ * occ_bg2
                warped_2 = warped_2[:, :, ::ds_scale, ::ds_scale]
            elif warp_method == 1:
                warped_1 = continuous_noise_fwd_warp(
                    high_res_noise_0, f_flow, alpha)
                warped_2 = continuous_noise_fwd_warp(
                    high_res_noise_1, b_flow, (1 - alpha))
            elif warp_method == 2:
                warped_1 = forward_flow_warp(latents[0].unsqueeze(
                    0), f_flow_ds * alpha)
                warped_2 = forward_flow_warp(
                    latents[-1].unsqueeze(0), b_flow_ds * (1-alpha))
            else:
                warped_1 = latents[0]
                warped_2 = latents[-1]

            if enable_interp:
                latents[i] = slerp(
                    warped_1[0], warped_2[0], alpha)
            else:
                latents[i] = warped_1[0]

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        def save_activations(latents) -> None:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat(
                    [latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)

                self.attn_state.set_timestep(t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False, **extra_step_kwargs)[0]

        self.attn_state.set_store_id(0)
        save_activations(latents[0].unsqueeze(0))
        self.attn_state.set_store_id(1)
        save_activations(latents[-1].unsqueeze(0))

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(
                self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - \
            num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        self.attn_state.to_load()
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        with self.progress_bar(total=num_inference_steps) as progress_bar:

            for i, t in enumerate(timesteps):

                if self.interrupt:
                    continue

                self.attn_state.set_timestep(t)

                noise_pred_list = []
                for f_i in range(num_frames):
                    alpha = f_i / (num_frames - 1)
                    self.attn_state.set_alpha(alpha)

                    tmp_input = latents[f_i:f_i+1]

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat(
                        [tmp_input] * 2) if self.do_classifier_free_guidance else tmp_input
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t)

                    noise_pred_sample = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred_sample.chunk(
                            2)
                        noise_pred_sample = noise_pred_uncond + self.guidance_scale * \
                            (noise_pred_text - noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred_sample = rescale_noise_cfg(
                            noise_pred_sample, noise_pred_text, guidance_rescale=self.guidance_rescale)

                    noise_pred_list.append(noise_pred_sample)
                noise_pred = torch.cat(noise_pred_list)

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs,
                    return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop(
                        "prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            images = []
            for i in range(latents.shape[0]):

                image = self.vae.decode(latents[i:i+1] / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                    0
                ]
                images.append(image)
            image = torch.cat(images)
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
