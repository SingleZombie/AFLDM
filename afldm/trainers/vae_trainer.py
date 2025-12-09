import os
from dataclasses import dataclass

from diffusers import AutoencoderKL
from diffusers.training_utils import EMAModel
from afldm.models.vae_training import AutoencoderKLTraining
from diffusers.optimization import get_scheduler
import numpy as np
import torch
import torch.nn.functional as F
import lpips

from .trainer import Trainer
from afldm.shift_utils.shifters import ImageShifter, gen_random_offset
from afldm.shift_utils.metrics import mask_mse, psnr
from afldm.models.discriminator import Discriminator


def calculate_adaptive_weight(rec_loss, g_loss, last_layer):
    nll_grads = torch.autograd.grad(
        rec_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(
        g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight


@dataclass
class VAETrainingConfig:
    # Diffuion Models
    model_cfg: str
    pretrained_model_name_or_path: str = None
    use_disc: bool = False
    disc_cfg: str = None

    use_shift_loss: bool = False

    disc_weight: float = 1.0
    perceptual_weight: float = 1.0
    kl_weight: float = 1e-6
    max_grad_norm: float = 1.0

    # AdamW
    scale_lr: bool = False
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    gradient_accumulation_steps: int = 2

    # LR Scheduler
    lr_scheduler: str = 'constant'
    lr_warmup_steps: int = 500

    # EMA
    use_ema: bool = False
    foreach_ema: bool = False
    offload_ema: bool = False


class VAETrainer(Trainer):
    def __init__(self, weight_dtype, accelerator, logger, cfg: VAETrainingConfig):
        super().__init__(weight_dtype, accelerator, logger, cfg)

    def init_modules(self,
                     enable_xformer=False,
                     gradient_checkpointing=False):
        cfg = self.cfg
        # Load scheduler, tokenizer and models.
        model = AutoencoderKLTraining.from_config(
            AutoencoderKLTraining.load_config(cfg.model_cfg))
        if cfg.pretrained_model_name_or_path is not None:
            tmp_model = AutoencoderKL.from_pretrained(
                cfg.pretrained_model_name_or_path)
            model_dict = tmp_model.state_dict()
            model.load_state_dict(model_dict)

        model.train()
        self.model = model

        if cfg.use_ema:
            self.ema_model = EMAModel(
                model.parameters(),
                model_cls=AutoencoderKLTraining,
                model_config=model.config,
                foreach=cfg.foreach_ema,
            )
            self.ema_model.to(self.accelerator.device)

        if cfg.use_disc:
            discriminator = Discriminator.from_config(
                Discriminator.load_config(cfg.disc_cfg))
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
            self.discriminator = discriminator
            self.discriminator.to(self.accelerator.device)
        self.model.to(self.accelerator.device)

        if enable_xformer:
            self.model.enable_xformers_memory_efficient_attention()

        if gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        self.perceptual_loss_fn = lpips.LPIPS(
            net='vgg').to(self.accelerator.device)

    def init_optimizers(self, train_batch_size):
        if self.cfg.scale_lr:
            self.cfg.learning_rate = (
                self.cfg.learning_rate * self.cfg.gradient_accumulation_steps *
                train_batch_size * self.accelerator.num_processes
            )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            weight_decay=self.cfg.adam_weight_decay,
            eps=self.cfg.adam_epsilon,
        )
        if self.cfg.use_disc:
            self.disc_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
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
        if self.cfg.use_disc:
            self.model, self.discriminator, self.optimizer, self.disc_optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.discriminator, self.optimizer, self.disc_optimizer, self.train_dataloader, self.lr_scheduler
            )
        else:
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
            )

        if self.cfg.use_shift_loss:
            self.img_shifter = ImageShifter('ideal_crop', 1)
            self.latent_shifter = ImageShifter(
                'ideal_crop', self.model.downsample_ratio)

    def models_to_train(self):
        self.model.train()
        if self.cfg.use_disc:
            self.discriminator.train()

    def training_step(self, global_step, batch) -> dict:

        input_batch = batch["input"].to(dtype=self.weight_dtype)
        bsz = input_batch.shape[0]

        is_generator_step = not self.cfg.use_disc or (
            global_step // self.cfg.gradient_accumulation_steps) % 2 == 0

        shift_loss = 0
        log_dict = {}
        if is_generator_step:
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad(set_to_none=True)
                latent_dist = self.model(input_batch, True).latent_dist
                latents = latent_dist.sample()
                recon_input_batch = self.model(latents, False).sample
                _, _, h, w = recon_input_batch.shape

                mse_loss = F.mse_loss(input_batch.float(),
                                      recon_input_batch.float(),
                                      reduction="mean")
                perceptual_loss = self.perceptual_loss_fn(input_batch.float(),
                                                          recon_input_batch.float())
                perceptual_loss = torch.sum(perceptual_loss) / bsz

                if self.cfg.use_shift_loss:
                    ti, tj = gen_random_offset(
                        h * 0.75 // 2, w * 0.75 // 2, True, 1)
                    with torch.no_grad():
                        f_x = latents.detach()
                        t_f_x, mask = self.latent_shifter.shift(
                            f_x, ti / 8, tj / 8)

                        t_x, _ = self.img_shifter.shift(
                            input_batch, ti, tj)

                    f_t_x = self.model(t_x, True).latent_dist.sample()
                    enc_loss = mask_mse(f_t_x, t_f_x, mask)

                    # Decoder shift loss
                    with torch.no_grad():
                        f_x = recon_input_batch.detach()
                        t_f_x, mask = self.img_shifter.shift(f_x, ti, tj)

                        t_x, _ = self.latent_shifter.shift(
                            latents.detach(), ti / 8, tj / 8)

                    f_t_x = self.model(t_x, False).sample
                    dec_loss = mask_mse(f_t_x, t_f_x, mask)
                    shift_loss = (enc_loss + dec_loss)

                if self.cfg.use_disc:
                    disc_loss = -self.discriminator(recon_input_batch).mean()
                    last_dec_layer = self.accelerator.unwrap_model(
                        self.model).decoder.conv_out.weight
                    disc_weight = calculate_adaptive_weight(
                        mse_loss + perceptual_loss, disc_loss,
                        last_dec_layer) * self.cfg.disc_weight
                else:
                    disc_weight = disc_loss = torch.tensor(0)

                kl_loss = torch.sum(latent_dist.kl()) / bsz

                log_dict['mse_loss'] = mse_loss.item()
                log_dict['shift_loss'] = shift_loss.item()
                loss = mse_loss + shift_loss + \
                    self.cfg.perceptual_weight * perceptual_loss + \
                    self.cfg.kl_weight * kl_loss + \
                    disc_weight * disc_loss

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    params_to_clip = self.model.parameters()
                    self.accelerator.clip_grad_norm_(
                        params_to_clip, self.cfg.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
        else:
            with self.accelerator.accumulate(self.discriminator):
                self.disc_optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    # encode
                    latent_dist = self.model(input_batch, True).latent_dist

                    latents = latent_dist.sample()

                    # decode
                    recon_input_batch = self.model(latents, False).sample
                    recon_input_batch.detach_()
                real = self.discriminator(input_batch)
                fake = self.discriminator(recon_input_batch)
                loss = torch.mean(F.relu(1 + fake) +
                                  F.relu(1 - real)) * 0.5

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    params_to_clip = self.discriminator.parameters()
                    self.accelerator.clip_grad_norm_(
                        params_to_clip, self.cfg.max_grad_norm)
                self.disc_optimizer.step()
                self.lr_scheduler.step()

        if self.accelerator.sync_gradients:
            if is_generator_step:
                log_dict['train_loss'] = loss.item()
            else:
                log_dict['train_loss_disc'] = loss.item()
            if self.cfg.use_ema:
                self.ema_model.step(self.model.parameters())

        return log_dict

    @torch.no_grad()
    def validate(self, global_step):

        if self.cfg.use_ema:
            self.ema_model.store(self.model.parameters())
            self.ema_model.copy_to(self.model.parameters())

        if self.cfg.use_ema:
            self.ema_model.restore(self.model.parameters())

        tmp_model = self.accelerator.unwrap_model(self.model)
        tmp_model.eval()

        input_images = [self.dataset[id]['image']
                        for id in range(5)]
        input_images = torch.stack(input_images).to(self.accelerator.device)

        latent_dist = tmp_model(input_images, True).latent_dist
        latents = latent_dist.sample()
        reconstruct_images = tmp_model(latents, False).sample

        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                concat_imgs = torch.cat([input_images, reconstruct_images], 2)
                image = concat_imgs.cpu()
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()
                np_images = np.stack([np.asarray(img)
                                     for img in image])
                tracker.writer.add_images(
                    'validation', np_images, global_step, dataformats="NHWC")

        if self.valid_dataloader is not None:
            tot_mse_loss = 0
            tot_perceptual_loss = 0
            tot_psnr = 0

            for x in self.valid_dataloader:

                x = x['input'].to(self.accelerator.device)
                batch_size = x.shape[0]
                recon_x, _ = tmp_model(x, return_dict=False)
                mse_loss = (F.mse_loss(x, recon_x) * batch_size).item()
                perceptual_loss = (self.perceptual_loss_fn(
                    x, recon_x).mean() * batch_size).item()
                tot_mse_loss += mse_loss
                tot_perceptual_loss += perceptual_loss
                tot_psnr += psnr(x, recon_x).item()

            tot_len = len(self.valid_dataset)
            tot_mse_loss /= tot_len
            tot_perceptual_loss /= tot_len
            tot_psnr = tot_psnr / tot_len

            msg_dict = {'mse': tot_mse_loss, 'perceptual': tot_perceptual_loss,
                        'psnr': tot_psnr}
            self.accelerator.log(msg_dict, step=global_step)

    def save_pipeline(self, output_dir):
        model = self.accelerator.unwrap_model(self.model)
        if self.cfg.use_ema:
            self.ema_model.copy_to(model.parameters())

        model.save_pretrained(output_dir)

    def save_model_hook(self, models, weights, output_dir):
        if self.accelerator.is_main_process:
            if self.cfg.use_ema:
                self.ema_model.save_pretrained(
                    os.path.join(output_dir, "model_ema"))

            for i, model in enumerate(models):
                if i == 0:
                    model.save_pretrained(os.path.join(output_dir, "vae"))
                elif i == 1:  # discriminator
                    model.save_pretrained(os.path.join(
                        output_dir, "discriminator"))
                weights.pop()

    def load_model_hook(self, models, input_dir):
        if self.cfg.use_ema:
            load_model = EMAModel.from_pretrained(
                os.path.join(input_dir, "model_ema"), AutoencoderKLTraining, foreach=self.cfg.foreach_ema
            )
            self.ema_model.load_state_dict(load_model.state_dict())
            if self.cfg.offload_ema:
                self.ema_model.pin_memory()
            else:
                self.ema_model.to(self.accelerator.device)
            del load_model

        if self.cfg.use_disc:
            discriminator = models.pop()
            load_model = Discriminator.from_pretrained(
                input_dir, subfolder="discriminator")
            discriminator.register_to_config(**load_model.config)
            discriminator.load_state_dict(load_model.state_dict())

        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = AutoencoderKLTraining.from_pretrained(
                input_dir, subfolder="vae")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            self.model = model
            del load_model
