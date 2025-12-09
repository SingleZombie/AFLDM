import inspect
import os
from dataclasses import dataclass

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from afldm.models.af_vae import AliasFreeAutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from .trainer import Trainer
from afldm.shift_utils.shifters import gen_random_offset, ImageShifter, gen_valid_mask
from afldm.pipelines.cross_frame_attn import CrossFrameAttnProcessor, AttnState
from afldm.shift_utils.metrics import mask_mse
# from afldm.models.fir_resblock import mod_unet, mod_vae_all


@dataclass
class SDTextTrainingConfig:
    # Diffuion Models
    pretrained_model_name_or_path: str
    revision: str = None
    variant: str = None
    ddpm_num_steps: int = 1000
    ddpm_beta_schedule: str = 'linear'
    prediction_type: str = 'epsilon'
    ddpm_num_inference_steps: int = 100
    max_grad_norm: float = 1.0
    use_shift_loss: bool = False

    # Validation
    valid_seed: int = 0
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
    offload_ema: bool = False


def log_validation(
    pipeline,
    seed,
    num_validation_images,
    accelerator,
    global_step,
    is_final_validation=False,
):
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if seed is not None:
        generator = generator.manual_seed(seed)
    images = []

    # autocast_ctx = torch.autocast(accelerator.device.type)

    # with autocast_ctx:
    for _ in range(num_validation_images):
        images.append(pipeline("forest",
                               num_inference_steps=20,
                               guidance_scale=7.5,
                               generator=generator).images[0])

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                phase_name, np_images, global_step, dataformats="NHWC")
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


class SDTextTrainer(Trainer):
    def __init__(self, weight_dtype, accelerator, logger, cfg: SDTextTrainingConfig):
        super().__init__(weight_dtype, accelerator, logger, cfg)

    def init_modules(self,
                     enable_xformer=False,
                     gradient_checkpointing=False):
        cfg = self.cfg
        # Load scheduler, tokenizer and models.
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="tokenizer", revision=cfg.revision
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="text_encoder", revision=cfg.revision
        )

        self.vae = AliasFreeAutoencoderKL.from_pretrained(
            'pretrained/vae_abl4_shift')
        # self.vae = AutoencoderKL.from_pretrained(
        #     cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision, variant=cfg.variant
        # )

        self.unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision, variant=cfg.variant
        )
        # freeze parameters of models to save more memory
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.train()

        # for name, module in self.unet.named_modules():
        #     if hasattr(module, "attn2"):
        #         for param in module.attn2.parameters():
        #             param.requires_grad = False

        mod_unet(self.unet, True, True)

        if cfg.use_ema:
            ema_unet = UNet2DConditionModel.from_pretrained(
                cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision, variant=cfg.variant
            )
            ema_unet = EMAModel(
                ema_unet.parameters(),
                model_cls=UNet2DConditionModel,
                model_config=ema_unet.config,
                foreach=cfg.foreach_ema,
            )
            self.ema_unet = ema_unet
            self.ema_unet.to(self.accelerator.device, dtype=self.weight_dtype)

        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        if enable_xformer:
            self.unet.enable_xformers_memory_efficient_attention()

        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

    def get_tokenizer(self):
        return self.tokenizer

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

    def prepare_modules(self):
        self.attn_state = AttnState()
        attn_processor_dict = {}
        if self.cfg.use_shift_loss:
            for k in self.unet.attn_processors.keys():
                attn_processor_dict[k] = CrossFrameAttnProcessor(
                    self.attn_state)
            self.unet.set_attn_processor(attn_processor_dict)
        self.shifter = ImageShifter('ideal', 8)

        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

    def models_to_train(self):
        self.unet.train()

    def training_step(self, global_step, batch) -> dict:
        train_loss = 0.0
        shift_loss = 0.0
        with self.accelerator.accumulate(self.unet):
            with torch.no_grad():
                # Convert images to latent space
                latents = self.vae.encode(batch["pixel_values"].to(
                    dtype=self.weight_dtype)).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                input_ids = batch["input_ids"]

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(
                latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(
                input_ids, return_dict=False)[0]

            ti, tj = gen_random_offset(int(256*0.75), int(256*0.75), True, 1)
            ti /= 8
            tj /= 8

            self.attn_state.reset()
            model_pred_0 = self.unet(noisy_latents, timesteps,
                                     encoder_hidden_states).sample

            self.attn_state.to_load()

            # Get the target for loss depending on the prediction type
            if self.cfg.prediction_type is not None:
                # set prediction_type of scheduler if defined
                self.noise_scheduler.register_to_config(
                    prediction_type=self.cfg.prediction_type)

            if self.cfg.use_shift_loss:
                mask = gen_valid_mask(noisy_latents.shape, ti,
                                      tj).to(noisy_latents.device)

                shifted_noisy_latents, _ = self.shifter.translate(
                    noisy_latents, ti, tj)
                target2, _ = self.shifter.translate(model_pred_0, ti, tj)

                target = target2

                model_pred = self.unet(shifted_noisy_latents, timesteps,
                                       encoder_hidden_states).sample

                shift_loss = mask_mse(model_pred.float(),
                                      target.float(), mask)

            ori_loss = F.mse_loss(model_pred_0.float(),
                                  noise.float(), reduction="mean")

            mse_loss = shift_loss + ori_loss

            loss = mse_loss

            train_batch_size = latents.shape[0]

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = self.accelerator.gather(
                loss.repeat(train_batch_size)).mean()
            train_loss += avg_loss.item() / self.accelerator.gradient_accumulation_steps

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
            logs = {"train_loss": train_loss}
            if self.cfg.use_ema:
                self.ema_unet.step(self.unet.parameters())

        return logs

    def validate(self, global_step):
        if self.cfg.use_ema:
            self.ema_unet.store(self.unet.parameters())
            self.ema_unet.copy_to(self.unet.parameters())
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            text_encoder=self.text_encoder,
            vae=self.vae.to(torch.float32),
            unet=self.accelerator.unwrap_model(self.unet),
            revision=self.cfg.revision,
            variant=self.cfg.variant,
        )
        pipeline.torch_dtype = self.weight_dtype

        self.attn_state.reset()
        log_validation(
            pipeline, self.cfg.valid_seed, self.cfg.valid_batch_size, self.accelerator, global_step)
        if self.cfg.use_ema:
            self.ema_unet.restore(self.unet.parameters())

        self.vae.to(self.weight_dtype)

        del pipeline
        torch.cuda.empty_cache()

    def save_pipeline(self, output_dir):
        unet = self.accelerator.unwrap_model(self.unet)
        if self.cfg.use_ema:
            self.ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=unet,
            revision=self.cfg.revision,
            variant=self.cfg.variant,
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
                os.path.join(input_dir, "unet_ema"), UNet2DConditionModel, foreach=self.cfg.foreach_ema
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
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model
