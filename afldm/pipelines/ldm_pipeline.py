from diffusers.models import UNet2DModel, AutoencoderKL
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
import inspect
from typing import List, Optional, Tuple, Union

from tqdm import tqdm
import torch


class MyLDMPipeline(DiffusionPipeline):
    r"""
    Pipeline for unconditional image generation using latent diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        vae ([`AutoencoderKL`]):
            VAE to encode and decode images to and from latent representations.
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            [`DDIMScheduler`] is used in combination with `unet` to denoise the encoded image latents.
    """

    def __init__(self, vae: AutoencoderKL, unet: UNet2DModel, scheduler: DDIMScheduler):
        super().__init__()
        self.register_modules(vae=vae, unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        latents=None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import LDMPipeline

        >>> # load model and scheduler
        >>> pipe = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        self.scheduler = DDIMScheduler.from_config(self.scheduler.config)

        if latents is None:
            latents = randn_tensor(
                (batch_size, self.unet.config.in_channels,
                 self.unet.config.sample_size, self.unet.config.sample_size),
                generator=generator,
            )
        latents = latents.to(device=self.device, dtype=self.unet.dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(
            self.scheduler.step).parameters.keys())

        extra_kwargs = {"generator": generator}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(self.scheduler.timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            # predict the noise residual
            noise_prediction = self.unet(latent_model_input, t).sample
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_prediction, t, latents, **extra_kwargs).prev_sample

        if output_type == 'latent':
            return latents

        latents = latents.to(self.vae.dtype)

        # adjust latents with inverse of vae scale
        latents = latents / self.vae.config.scaling_factor
        # decode the image latents with the VAE
        image = self.vae.decode(latents).sample

        if output_type != 'pt':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            if output_type == "pil":
                image = self.numpy_to_pil(image)

            if not return_dict:
                return (image,)

            return ImagePipelineOutput(images=image)
        return image

    @torch.no_grad()
    def ddim_inversion(self, latent, bar=True):
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            if bar:
                iterator = tqdm(timesteps, desc="DDIM inversion")
            else:
                iterator = timesteps
            for i, t in enumerate(iterator):

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                input_latent = latent
                eps = self.unet(input_latent, t).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps

        return latent
