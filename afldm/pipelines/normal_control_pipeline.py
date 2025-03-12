from diffusers import StableDiffusionControlNetPipeline
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from afldm.pipelines.cross_frame_attn import CrossFrameAttnProcessor, AttnState
from afldm.shift_utils.shifters import ImageShifter, FilterType, image_random_translate, gen_random_offset
from afldm.shift_utils.metrics import mask_psnr, mask_mse
from torchvision import transforms
import cv2


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
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
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(
            scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class NormControlPipeline(StableDiffusionControlNetPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
        is_yoso=True
    ):
        self.is_yoso = is_yoso
        super().__init__(vae, text_encoder,
                         tokenizer, unet, controlnet, scheduler,
                         safety_checker, feature_extractor, image_encoder, requires_safety_checker)

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None],
                  PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        shift_latent=False,
        use_cfa=False,
        num_frames=11,
        horizontal_only=True,
        zero_input=True,
        return_psnr=False,
        ** kwargs,
    ):
        # if isinstance(image, list):
        #     image_batch_size = len(image)
        # else:
        #     image_batch_size = image.shape[0]
        # prompt = [""] * image_batch_size
        prompt = ''
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

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        controlnet = self.controlnet._orig_mod if is_compiled_module(
            self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(
                control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(
                control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(
                controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     image,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     ip_adapter_image,
        #     ip_adapter_image_embeds,
        #     controlnet_conditioning_scale,
        #     control_guidance_start,
        #     control_guidance_end,
        #     callback_on_step_end_tensor_inputs,
        # )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [
                controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get(
                "scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare image
        image = self.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            guess_mode=guess_mode,
        )
        height, width = image.shape[-2:]

        img_shifter = ImageShifter('ideal', 1)
        latent_shifter = ImageShifter('ideal', 8)

        if shift_latent:
            image = self.vae.encode(image).latent_dist.sample()
            image = image * self.vae.config.scaling_factor
            image = image.repeat(num_frames, 1, 1, 1)
            for i in range(1, num_frames):
                tj = i / 8 * 4
                if horizontal_only:
                    ti = 0
                else:
                    ti = tj
                image[i:i+1], _ = latent_shifter.translate(image[0:1], ti, tj)

        else:
            image = image.repeat(num_frames, 1, 1, 1)

            for i in range(1, num_frames):
                tj = i * 4
                if horizontal_only:
                    ti = 0
                else:
                    ti = tj
                image[i:i+1], _ = img_shifter.translate(image[0:1], ti, tj)
                image[i:i+1, :, :, :tj] = 0

            image = self.vae.encode(image).latent_dist.sample()
            image = image * self.vae.config.scaling_factor

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        if self.is_yoso:
            if zero_input:
                latents = torch.zeros_like(latents)
            else:
                latents = torch.randn_like(latents)

        attn_state = AttnState()
        if use_cfa:
            attn_processor_dict = {}
            for k in self.unet.attn_processors.keys():
                attn_processor_dict[k] = CrossFrameAttnProcessor(
                    attn_state)
            self.unet.set_attn_processor(attn_processor_dict)

            attn_processor_dict = {}
            for k in self.controlnet.attn_processors.keys():
                attn_processor_dict[k] = CrossFrameAttnProcessor(
                    attn_state)
            self.controlnet.set_attn_processor(attn_processor_dict)

        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(
                self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        # Relevant thread:
        # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
        if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
            torch._inductor.cudagraph_mark_step_begin()
        # expand the latents if we are doing classifier free guidance

        output = torch.empty_like(image)
        default_t = self.scheduler.config.num_train_timesteps - 1

        def denoise_output(latents, image, t):
            latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            attn_state.set_timestep(t)

            # controlnet(s) inference
            if guess_mode and self.do_classifier_free_guidance:
                # Infer ControlNet only for the conditional batch.
                control_model_input = latents
                control_model_input = self.scheduler.scale_model_input(
                    control_model_input, t)
                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
            else:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input,
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=image,
                conditioning_scale=1.0,
                guess_mode=guess_mode,
                return_dict=False,
            )

            # predict the noise residual
            output = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

            return output

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        def denoise(latents, image):
            if self.is_yoso:
                return denoise_output(latents, image, default_t)
            else:
                for i, t in enumerate(timesteps):
                    noise_pred = denoise_output(latents, image, t)
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                return latents

        attn_state.reset()
        output[0:1] = denoise(latents, image[0:1])
        attn_state.to_load()
        for i in range(1, output.shape[0]):
            output[i:i+1] = denoise(latents, image[i:i+1])
        latents = output

        has_nsfw_concept = None
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
        else:
            image = latents
            has_nsfw_concept = None

        first_img = image[0:1]
        gt_outputs = torch.empty_like(image[1:])
        sum_psnr = 0
        for i in range(1, num_frames):
            tj = i
            if horizontal_only:
                ti = 0
            else:
                ti = tj
            gt_outputs[i - 1], mask = img_shifter.translate(first_img, ti, tj)
            shift_psnr = mask_psnr(gt_outputs[i - 1], image[i], mask)
            image[i:i+1] *= mask
            sum_psnr += shift_psnr
        if num_frames > 1:
            avg_psnr = sum_psnr / (num_frames - 1)
            # print(avg_psnr)

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            if return_psnr:
                return (image, has_nsfw_concept), avg_psnr
            else:
                return (image, has_nsfw_concept)

        if return_psnr:
            return (StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), avg_psnr)
        else:
            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
