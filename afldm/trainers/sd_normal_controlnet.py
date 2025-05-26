import inspect
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


from diffusers import ControlNetModel, DDPMScheduler, DiffusionPipeline, StableDiffusionControlNetPipeline, UNet2DConditionModel, ModelMixin
from afldm.models.af_vae import AliasFreeAutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.configuration_utils import register_to_config
from diffusers.models.controlnet import ControlNetOutput

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from .trainer import Trainer
from afldm.shift_utils.shifters import gen_random_offset, ImageShifter, gen_valid_mask
from afldm.pipelines.cross_frame_attn import CrossFrameAttnProcessor, AttnState
from afldm.shift_utils.metrics import mask_mse
from afldm.pipelines.normal_control_pipeline import NormControlPipeline


@dataclass
class NormControlNetConfig:
    # Diffuion Models
    pretrained_model_name_or_path: str
    revision: str = None
    variant: str = None
    ddpm_num_steps: int = 1000
    ddpm_beta_schedule: str = 'linear'
    prediction_type: str = 'epsilon'
    ddpm_num_inference_steps: int = 100
    is_yoso: bool = True

    unet_path: str = None
    controlnet_path: str = None

    max_grad_norm: float = 1.0
    alias_free: bool = True
    cross_frame_attn: bool = False

    # Validation
    valid_seed: int = 0
    valid_batch_size: int = 1

    # AdamW
    scale_lr: bool = False
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


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class MyControlNetModel(ControlNetModel, ModelMixin):
    @register_to_config
    def __init__(
            self,
            in_channels: int = 4,
            conditioning_channels: int = 3,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str, ...] = (
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
            only_cross_attention: Union[bool, Tuple[bool]] = False,
            block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
            layers_per_block: int = 2,
            downsample_padding: int = 1,
            mid_block_scale_factor: float = 1,
            act_fn: str = "silu",
            norm_num_groups: Optional[int] = 32,
            norm_eps: float = 1e-5,
            cross_attention_dim: int = 1280,
            transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
            encoder_hid_dim: Optional[int] = None,
            encoder_hid_dim_type: Optional[str] = None,
            attention_head_dim: Union[int, Tuple[int, ...]] = 8,
            num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
            use_linear_projection: bool = False,
            class_embed_type: Optional[str] = None,
            addition_embed_type: Optional[str] = None,
            addition_time_embed_dim: Optional[int] = None,
            num_class_embeds: Optional[int] = None,
            upcast_attention: bool = False,
            resnet_time_scale_shift: str = "default",
            projection_class_embeddings_input_dim: Optional[int] = None,
            controlnet_conditioning_channel_order: str = "rgb",
            conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (
                16, 32, 96, 256),
            global_pool_conditions: bool = False,
            addition_embed_type_num_heads: int = 64):
        super().__init__(in_channels, conditioning_channels, flip_sin_to_cos, freq_shift, down_block_types, mid_block_type, only_cross_attention, block_out_channels, layers_per_block, downsample_padding, mid_block_scale_factor, act_fn, norm_num_groups, norm_eps, cross_attention_dim, transformer_layers_per_block, encoder_hid_dim, encoder_hid_dim_type,
                         attention_head_dim, num_attention_heads, use_linear_projection, class_embed_type, addition_embed_type, addition_time_embed_dim, num_class_embeds, upcast_attention, resnet_time_scale_shift, projection_class_embeddings_input_dim, controlnet_conditioning_channel_order, conditioning_embedding_out_channels, global_pool_conditions, addition_embed_type_num_heads)
        self.controlnet_cond_embedding = nn.Identity()
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in2 = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )
        zero_module(self.conv_in2)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        # check channel order
        channel_order = self.config.controlnet_conditioning_channel_order

        if channel_order == "rgb":
            # in rgb order by default
            ...
        elif channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        else:
            raise ValueError(
                f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor(
                [timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)

            elif self.config.addition_embed_type == "text_time":
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                    )
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb if aug_emb is not None else emb

        # 2. pre-process
        sample = self.conv_in(sample)
        controlnet_cond = self.conv_in2(controlnet_cond)

        sample = sample + controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb)

        # 5. Control net blocks
        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + \
                (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        if guess_mode and not self.config.global_pool_conditions:
            # 0.1 to 1.0
            scales = torch.logspace(-1, 0, len(down_block_res_samples) +
                                    1, device=sample.device)
            scales = scales * conditioning_scale
            down_block_res_samples = [
                sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * \
                scales[-1]  # last one
        else:
            down_block_res_samples = [
                sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(
                mid_block_res_sample, dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )


def log_validation(
    pipe,
    use_cfa,
    gt_images,
    cond_images,
    accelerator,
    global_step,
    is_final_validation=False,
):
    with torch.no_grad():
        res = pipe(cond_images,
                   use_cfa=use_cfa,
                   num_frames=1,
                   output_type='pt').images

    cond_images = (cond_images + 1) / 2
    gt_images = (gt_images + 1) / 2
    psnr_cal = PeakSignalNoiseRatio()

    mse = F.mse_loss(gt_images, res).item()
    psnr_cal.update(gt_images, res)
    psnr = psnr_cal.compute().item()

    res_cat = torch.cat((cond_images, gt_images, res), 2)
    images = res_cat.cpu().permute(0, 2, 3, 1).numpy()

    accelerator.log({'mse': mse, 'psnr': psnr}, step=global_step)

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


class NormControlNetTrainer(Trainer):
    def __init__(self, weight_dtype, accelerator, logger, cfg):
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

        # self.vae = AliasFreeAutoencoderKL.from_pretrained(
        #     cfg.pretrained_model_name_or_path, subfolder="vae")

        if cfg.alias_free:

            self.vae = AliasFreeAutoencoderKL.from_pretrained(
                'pretrained/vae_final_shift2_fix')

            if self.cfg.unet_path is not None:
                self.unet = UNet2DConditionModel.from_pretrained(
                    self.cfg.unet_path
                )
            else:
                self.unet = UNet2DConditionModel.from_pretrained(
                    'train_ckpt/laion_unet_finalshift2_fix_vae_2_1e-6/checkpoint-142000/unet'
                )

            if self.cfg.controlnet_path is not None:
                self.controlnet = MyControlNetModel.from_pretrained(
                    self.cfg.controlnet_path)
            else:
                self.controlnet = MyControlNetModel.from_unet(
                    self.unet, conditioning_channels=4)

            mod_unet(self.unet, True, True)
            mod_controlnet(self.controlnet, True, True)
        else:
            self.vae = AliasFreeAutoencoderKL.from_pretrained(
                cfg.pretrained_model_name_or_path, subfolder="vae")

            if self.cfg.unet_path is not None:
                self.unet = UNet2DConditionModel.from_pretrained(
                    self.cfg.unet_path
                )
            else:
                self.unet = UNet2DConditionModel.from_pretrained(
                    cfg.pretrained_model_name_or_path, subfolder="unet")

            if self.cfg.controlnet_path is not None:
                self.controlnet = MyControlNetModel.from_pretrained(
                    self.cfg.controlnet_path)
            else:
                self.controlnet = MyControlNetModel.from_unet(
                    self.unet, conditioning_channels=4)

        # freeze parameters of models to save more memory
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.down_blocks.requires_grad_(False)
        self.unet.mid_block.requires_grad_(False)

        self.unet.train()
        self.controlnet.train()

        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.controlnet.to(self.accelerator.device, dtype=self.weight_dtype)

        if enable_xformer:
            self.unet.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()

        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            self.controlnet.enable_gradient_checkpointing()

        empty_ids = self.tokenizer('', padding="max_length", truncation=True,
                                   max_length=self.tokenizer.model_max_length, return_tensors="pt")
        self.empty_text_emb = self.text_encoder(
            empty_ids.input_ids.to(self.text_encoder.device), return_dict=False)[0]

    def get_tokenizer(self):
        return self.tokenizer

    def init_optimizers(self, train_batch_size):
        if self.cfg.scale_lr:
            self.cfg.learning_rate = (
                self.cfg.learning_rate * self.cfg.gradient_accumulation_steps *
                train_batch_size * self.accelerator.num_processes
            )

        all_parameters = list(
            self.unet.up_blocks.parameters()) + list(self.controlnet.parameters()) \
            + list(self.unet.conv_norm_out.parameters()) \
            + list(self.unet.conv_out.parameters())

        self.optimizer = torch.optim.AdamW(
            all_parameters,
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

    def set_dataset(self, dataset, train_dataloader, valid_dataset=None, valid_dataloader=None):
        super().set_dataset(dataset, train_dataloader, valid_dataset, valid_dataloader)
        valid_ids = np.random.choice(
            len(dataset), self.cfg.valid_batch_size, replace=False).tolist()

        self.gt_images = [dataset[id]["pixel_values"].unsqueeze(0)
                          for id in valid_ids]
        self.gt_images = torch.cat(self.gt_images).to(self.accelerator.device)

        self.cond_images = [dataset[id]["conditioning_pixel_values"].unsqueeze(0)
                            for id in valid_ids]
        self.cond_images = torch.cat(
            self.cond_images).to(self.accelerator.device)

    def prepare_modules(self):
        self.attn_state = AttnState()

        if self.cfg.alias_free:
            attn_processor_dict = {}
            for k in self.unet.attn_processors.keys():
                attn_processor_dict[k] = CrossFrameAttnProcessor(
                    self.attn_state)
            self.unet.set_attn_processor(attn_processor_dict)

            attn_processor_dict = {}
            for k in self.controlnet.attn_processors.keys():
                attn_processor_dict[k] = CrossFrameAttnProcessor(
                    self.attn_state)
            self.controlnet.set_attn_processor(attn_processor_dict)

        self.shifter = ImageShifter('ideal', 8)
        # self.image_shifter = ImageShifter('ideal', 1)

        self.unet, self.controlnet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.controlnet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

    def models_to_train(self):
        self.unet.train()

    def training_step(self, global_step, batch) -> dict:
        train_loss = 0.0
        shift_loss = 0.0
        with self.accelerator.accumulate(self.unet):
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

            if self.cfg.is_yoso:
                rand_number = torch.randint(0, 10, (1, ))[0]
                if rand_number < 4:
                    noisy_latents = torch.zeros_like(noise)
                else:
                    noisy_latents = noise
            else:
                noisy_latents = self.noise_scheduler.add_noise(
                    latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(
                input_ids, return_dict=False)[0]

            controlnet_image = batch["conditioning_pixel_values"].to(
                dtype=self.weight_dtype)

            controlnet_latents = self.vae.encode(
                controlnet_image).latent_dist.sample()
            controlnet_latents = controlnet_latents * self.vae.config.scaling_factor

            ti, tj = gen_random_offset(int(256*0.75), int(256*0.75), True, 1)
            ti /= 8
            tj /= 8

            self.attn_state.reset()
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_latents,
                return_dict=False,
            )
            model_pred_0 = self.unet(noisy_latents, timesteps,
                                     encoder_hidden_states,
                                     down_block_additional_residuals=[
                                         sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples
                                     ],
                                     mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),).sample

            self.attn_state.to_load()

            # Get the target for loss depending on the prediction type
            if self.cfg.prediction_type is not None:
                # set prediction_type of scheduler if defined
                self.noise_scheduler.register_to_config(
                    prediction_type=self.cfg.prediction_type)

            if self.cfg.alias_free:

                mask = gen_valid_mask(noisy_latents.shape, ti,
                                      tj).to(noisy_latents.device)

                shifted_noisy_latents, _ = self.shifter.translate(
                    noisy_latents, ti, tj)
                shifted_controlnet_latents, _ = self.shifter.translate(
                    controlnet_latents, ti, tj)
                target2, _ = self.shifter.translate(model_pred_0, ti, tj)

                target = target2

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    shifted_noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=shifted_controlnet_latents,
                    return_dict=False,
                )
                model_pred = self.unet(shifted_noisy_latents, timesteps,
                                       encoder_hidden_states,
                                       down_block_additional_residuals=[
                                           sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples
                                       ],
                                       mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),).sample

                shift_loss = mask_mse(model_pred.float(),
                                      target.float(), mask)

            # shifted_ori_loss = mask_mse(model_pred.float(),
            #                             target1.float(), mask)
            if self.cfg.is_yoso:
                ori_loss = F.mse_loss(model_pred_0.float(),
                                      latents.float(), reduction="mean")
            else:
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

        pipe = NormControlPipeline(
            self.vae,
            self.text_encoder,
            self.tokenizer,
            self.accelerator.unwrap_model(self.unet),
            self.accelerator.unwrap_model(self.controlnet),
            self.noise_scheduler,
            None,
            None,
            None,
            False,
            self.cfg.is_yoso
        )

        log_validation(
            pipe,
            self.cfg.alias_free,
            self.gt_images,
            self.cond_images,
            self.accelerator, global_step)

        if self.cfg.alias_free:
            attn_processor_dict = {}
            for k in self.unet.module.attn_processors.keys():
                attn_processor_dict[k] = CrossFrameAttnProcessor(
                    self.attn_state)
            self.unet.module.set_attn_processor(attn_processor_dict)

            attn_processor_dict = {}
            for k in self.controlnet.module.attn_processors.keys():
                attn_processor_dict[k] = CrossFrameAttnProcessor(
                    self.attn_state)
            self.controlnet.module.set_attn_processor(attn_processor_dict)

        if self.cfg.use_ema:
            self.ema_unet.restore(self.unet.parameters())

        torch.cuda.empty_cache()

    def save_pipeline(self, output_dir):
        unet = self.accelerator.unwrap_model(self.unet)
        controlnet = self.accelerator.unwrap_model(self.controlnet)
        if self.cfg.use_ema:
            self.ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=unet,
            controlnet=controlnet,
            revision=self.cfg.revision,
            variant=self.cfg.variant,
        )
        pipeline.save_pretrained(output_dir)

    def save_model_hook(self, models, weights, output_dir):
        if self.accelerator.is_main_process:

            models[0].save_pretrained(os.path.join(output_dir, "unet"))
            models[1].save_pretrained(os.path.join(output_dir, "controlnet"))

            weights.pop()
            weights.pop()

    def load_model_hook(self, models, input_dir):

        model = models.pop()
        load_model = MyControlNetModel.from_pretrained(
            input_dir, subfolder="controlnet")
        model.register_to_config(**load_model.config)
        model.load_state_dict(load_model.state_dict())
        del load_model

        model = models.pop()
        load_model = UNet2DConditionModel.from_pretrained(
            input_dir, subfolder="unet")
        model.register_to_config(**load_model.config)

        model.load_state_dict(load_model.state_dict())
        del load_model
