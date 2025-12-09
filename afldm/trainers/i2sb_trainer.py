import os
from dataclasses import dataclass

from diffusers import UNet2DModel
from afldm.models.af_vae import AliasFreeAutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import numpy as np
import torch
import torch.nn.functional as F


from torchmetrics import PeakSignalNoiseRatio

from .trainer import Trainer
from afldm.af_libs.superresolution import build_sr4x
from afldm.shift_utils.shifters import gen_random_offset, ImageShifter, gen_valid_mask
from ..pipelines.cross_frame_attn import (CrossFrameAttnProcessor, AttnState,
                                          get_unet_attn_processors,
                                          set_unet_attn_processor)
from ..pipelines.i2sb_pipeline import I2SBLDMPipeline
from ..shift_utils.metrics import mask_mse
from ..schedulers.i2sb_scheduler import I2SBScheduler
from afldm.af_modules.af_api import make_af_unet


class DegradTransform(torch.nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.sr_func = build_sr4x('cpu', 'bicubic', resolution)

    def forward(self, img):
        return self.sr_func(img)


@dataclass
class I2SBLDMTrainingConfig:
    # Diffuion Models
    scheduler_path: str
    vae_path: str = None
    unet_config: str = None
    unet_path: str = None
    af_models: bool = True
    is_ode: bool = True
    use_cfa: bool = False

    max_grad_norm = 1.0

    # Validation
    valid_seed = 0
    valid_batch_size: int = 1

    # AdamW
    scale_lr = False
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08

    # LR Scheduler
    lr_scheduler: str = 'constant'
    lr_warmup_steps: int = 500

    # EMA
    use_ema: bool = False
    foreach_ema: bool = False


def log_validation(
    pipeline,
    seed,
    hq_images,
    lq_images,
    accelerator,
    step,
    is_ode,
    is_final_validation=False,
):
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator()
    if seed is not None:
        generator = generator.manual_seed(seed)
    images = []

    hq_images = torch.cat(hq_images).to(accelerator.device)
    hq_images = (hq_images + 1) / 2
    lq_images = torch.cat(lq_images).to(accelerator.device)
    lq_images = (lq_images + 1) / 2
    res = pipeline(lq_images, generator=generator,
                   is_ode=is_ode, output_type='pt')
    res = res.clip(-1, 1)
    res = (res + 1) / 2

    psnr_cal = PeakSignalNoiseRatio().to(accelerator.device)

    mse = F.mse_loss(hq_images, res).item()
    psnr = psnr_cal(hq_images, res).item()

    res_cat = torch.cat((lq_images, hq_images, res), 2)
    images = res_cat.cpu().permute(0, 2, 3, 1).numpy()
    images = pipeline.numpy_to_pil(images)

    accelerator.log({'mse': mse, 'psnr': psnr}, step=step)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                phase_name, np_images, step, dataformats="NHWC")
        if tracker.name == "wandb":
            import wandb
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {''}") for i, image in enumerate(images)
                    ]
                }
            )
    return images


class I2SBTrainer(Trainer):
    def __init__(self, weight_dtype, accelerator, logger, cfg: I2SBLDMTrainingConfig):
        super().__init__(weight_dtype, accelerator, logger, cfg)

    def init_modules(self,
                     enable_xformer=False,
                     gradient_checkpointing=False):
        cfg = self.cfg
        # Load scheduler, tokenizer and models.
        self.noise_scheduler = I2SBScheduler.from_config(
            I2SBScheduler.load_config(cfg.scheduler_path)
        )

        self.vae = AliasFreeAutoencoderKL.from_pretrained(
            cfg.vae_path)
        if cfg.unet_path is not None:
            self.unet = UNet2DModel.from_pretrained(cfg.unet_path)
        else:
            self.unet = UNet2DModel.from_config(
                UNet2DModel.load_config(cfg.unet_config))
        # freeze parameters of models to save more memory
        self.vae.requires_grad_(False)
        self.unet.train()

        if cfg.af_models:
            make_af_unet(self.unet)

        if cfg.use_ema:
            ema_unet = UNet2DModel.from_config(
                UNet2DModel.load_config(cfg.unet_config))

            ema_unet = EMAModel(
                ema_unet.parameters(),
                model_cls=UNet2DModel,
                model_config=ema_unet.config,
                foreach=cfg.foreach_ema,
            )

        self.unet.to(self.accelerator.device)
        self.vae.to(self.accelerator.device)
        black_latent = torch.ones(1, 3, 256, 256).to(self.accelerator.device)
        black_latent = self.vae.encode(black_latent).latent_dist.mode()
        black_latent = black_latent * self.vae.config.scaling_factor
        self.black_latent = black_latent.to(dtype=self.weight_dtype)

        if enable_xformer:
            self.unet.enable_xformers_memory_efficient_attention()

        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

    def init_optimizers(self, train_batch_size):
        if self.cfg.scale_lr:
            self.cfg.learning_rate = (
                self.cfg.learning_rate * self.cfg.gradient_accumulation_steps *
                train_batch_size * self.accelerator.num_processes
            )
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.cfg.learning_rate,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            weight_decay=self.cfg.adam_weight_decay,
            eps=self.cfg.adam_epsilon,
        )

    def init_lr_schedulers(self, gradient_accumulation_steps, num_epochs):
        self.lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.lr_warmup_steps *
            gradient_accumulation_steps,
            num_training_steps=(len(self.train_dataloader)
                                * num_epochs)
        )

    def set_dataset(self, dataset, train_dataloader, valid_dataset=None,
                    valid_dataloader=None):
        super().set_dataset(dataset, train_dataloader, valid_dataset, valid_dataloader)
        valid_ids = np.random.choice(
            len(dataset), self.cfg.valid_batch_size, replace=False).tolist()

        if isinstance(dataset[0], torch.Tensor):
            degrad_func = build_sr4x(
                'cpu', 'bicubic', dataset[0].shape[2])

            self.hq_images = [dataset[id]
                              for id in valid_ids]

            self.lq_images = [degrad_func(x).clip(-1, 1)
                              for x in self.hq_images]
        else:
            degrad_func = build_sr4x(
                'cpu', 'bicubic', dataset[0]["image"].shape[2])

            self.hq_images = [dataset[id]["image"].unsqueeze(0)
                              for id in valid_ids]

            self.lq_images = [degrad_func(x).clip(-1, 1)
                              for x in self.hq_images]

    def prepare_modules(self):
        self.attn_state = AttnState()
        if self.cfg.use_cfa:
            attn_processor_dict = {}
            processors = get_unet_attn_processors(self.unet)
            for k in processors.keys():
                attn_processor_dict[k] = CrossFrameAttnProcessor(
                    self.attn_state)
            set_unet_attn_processor(self.unet, attn_processor_dict)
        self.shifter = ImageShifter('ideal', 8)
        self.img_shifter = ImageShifter('ideal', 1)

        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

    def models_to_train(self):
        self.unet.train()

    def training_step(self, global_step, batch) -> dict:
        shift_loss = 0.0
        with self.accelerator.accumulate(self.unet):
            # Convert images to latent space
            vae_input = batch["input"]
            bsz = vae_input.shape[0]

            degrad_func = build_sr4x(
                self.accelerator.device, 'bicubic', vae_input.shape[2])
            with torch.no_grad():

                latents = self.vae.encode(vae_input).latent_dist.mode()
                latents = latents * self.vae.config.scaling_factor
                latents = latents.to(dtype=self.weight_dtype)

                vae_lq_input = degrad_func(vae_input)
                latents_lq = self.vae.encode(vae_lq_input).latent_dist.mode()
                latents_lq = latents_lq * self.vae.config.scaling_factor
                latents_lq = latents_lq.to(dtype=self.weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            with torch.no_grad():
                noisy_latents = self.noise_scheduler.add_noise(
                    latents, latents_lq, timesteps, noise=noise, is_ode=self.cfg.is_ode)
                label = self.noise_scheduler.compute_label(
                    timesteps, latents, noisy_latents
                )

            ti, tj = gen_random_offset(int(128*0.75), int(128*0.75), True, 1)

            ti /= 8
            tj /= 8

            if self.cfg.use_cfa:
                self.attn_state.reset()
                model_pred_0 = self.unet(noisy_latents, timesteps).sample

                self.attn_state.to_load()
            else:
                model_pred_0 = self.unet(noisy_latents, timesteps).sample

            mask = gen_valid_mask(noisy_latents.shape, ti,
                                  tj).to(noisy_latents.device)

            shifted_noisy_latents, _ = self.shifter.shift(
                noisy_latents, ti, tj)

            if self.cfg.af_models:
                target2, _ = self.shifter.shift(model_pred_0, ti, tj)

                target = target2

                model_pred = self.unet(shifted_noisy_latents, timesteps).sample

                shift_loss = mask_mse(model_pred.float(),
                                      target.float(), mask)

            ori_loss = F.mse_loss(model_pred_0.float(),
                                  label.float(), reduction="mean")

            mse_loss = shift_loss + ori_loss

            loss = mse_loss

            # Backpropagate
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                params_to_clip = self.unet.parameters()
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.cfg.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        if self.accelerator.sync_gradients:
            logs = {"train_loss": loss.item()}
            if self.cfg.use_ema:
                self.ema_unet.step(self.unet.parameters())

        return logs

    def validate(self, global_step):
        if self.cfg.use_ema:
            self.ema_unet.store(self.unet.parameters())
            self.ema_unet.copy_to(self.unet.parameters())
        pipeline = I2SBLDMPipeline(
            vae=self.vae,
            unet=self.accelerator.unwrap_model(self.unet),
            scheduler=I2SBScheduler.from_config(self.noise_scheduler.config)
        ).to(torch.float32)
        self.attn_state.reset()
        log_validation(
            pipeline, self.cfg.valid_seed, self.hq_images, self.lq_images,
            self.accelerator, global_step, self.cfg.is_ode)
        if self.cfg.use_ema:
            self.ema_unet.restore(self.unet.parameters())

        del pipeline
        torch.cuda.empty_cache()

    def save_pipeline(self, output_dir):
        unet = self.accelerator.unwrap_model(self.unet)
        if self.cfg.use_ema:
            self.ema_unet.copy_to(unet.parameters())

        pipeline = I2SBLDMPipeline(
            vae=self.vae,
            unet=unet,
            scheduler=self.noise_scheduler
        )
        pipeline.save_pretrained(output_dir)

    def save_model_hook(self, models, weights, output_dir):
        if self.accelerator.is_main_process:
            if self.cfg.use_ema:
                self.ema_unet.save_pretrained(
                    os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(self, models, input_dir):
        if self.cfg.use_ema:
            load_model = EMAModel.from_pretrained(
                os.path.join(input_dir, "unet_ema"), UNet2DModel, foreach=self.cfg.foreach_ema
            )
            self.ema_unet.load_state_dict(load_model.state_dict())
            if self.cfg.offload_ema:
                self.ema_unet.pin_memory()
            else:
                self.ema_unet.to(self.accelerator.device)
            del load_model

        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DModel.from_pretrained(
                input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model
