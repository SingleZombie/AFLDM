from diffusers.models import UNet2DModel, AutoencoderKL
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.image_processor import VaeImageProcessor
import inspect
from typing import List, Optional, Tuple, Union

from tqdm import tqdm
import torch

from .ldm_pipeline import MyLDMPipeline
from ..schedulers.i2sb_scheduler import I2SBScheduler


class I2SBLDMPipeline(MyLDMPipeline):
    def __init__(self, vae: AutoencoderKL, unet: UNet2DModel, scheduler: I2SBScheduler):
        super().__init__(vae, unet, scheduler)
        self.image_processor = VaeImageProcessor()

    @torch.no_grad()
    def __call__(
        self,
        images,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        is_ode=False,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        device = self._execution_device

        H, W = images.shape[2:]

        img_list = [self.image_processor.preprocess(
            img, H, W).to(device) for img in images]
        img_tensor = torch.cat(img_list)

        latents = img_tensor.to(device=self.device, dtype=self.unet.dtype)
        latents = self.vae.encode(
            latents).latent_dist.sample() * self.vae.config.scaling_factor

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            if i == num_inference_steps - 1:
                break
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            # predict the noise residual
            noise_prediction = self.unet(latent_model_input, t).sample
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_prediction, t, latents,
                is_ode=is_ode, generator=generator).prev_sample

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
