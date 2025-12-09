from abc import ABCMeta, abstractmethod


class Trainer(metaclass=ABCMeta):
    def __init__(self, weight_dtype, accelerator, logger, cfg):
        self.weight_dtype = weight_dtype
        self.accelerator = accelerator
        self.logger = logger
        self.cfg = cfg

    @abstractmethod
    def init_modules(self,
                     enable_xformer: bool = False,
                     gradient_checkpointing: bool = False):
        pass

    @abstractmethod
    def init_optimizers(self, train_batch_size):
        pass

    @abstractmethod
    def init_lr_schedulers(self, gradient_accumulation_steps, num_epochs):
        pass

    def set_dataset(self, dataset,
                    train_dataloader,
                    valid_dataset=None,
                    valid_dataloader=None):
        self.dataset = dataset
        self.train_dataloader = train_dataloader
        self.valid_dataset = valid_dataset
        self.valid_dataloader = valid_dataloader

    @abstractmethod
    def prepare_modules(self):
        pass

    @abstractmethod
    def models_to_train(self):
        pass

    @abstractmethod
    def training_step(self, global_step, batch) -> dict:
        pass

    @abstractmethod
    def validate(self, global_step):
        pass

    @abstractmethod
    def save_pipeline(self):
        pass

    @abstractmethod
    def save_model_hook(self, models, weights, output_dir):
        pass

    @abstractmethod
    def load_model_hook(self, models, input_dir):
        pass


def create_trainer(type, weight_dtype, accelerator, logger, cfg) -> Trainer:
    from .vae_trainer import VAETrainer
    from .sd_text_trainer import SDTextTrainer
    from .ldm_trainer import LDMTrainer
    from .i2sb_trainer import I2SBTrainer
    from .sd_normal_controlnet import NormControlNetTrainer

    __TYPE_CLS_DICT = {
        'vae': VAETrainer,
        'sd_text': SDTextTrainer,
        'ldm': LDMTrainer,
        'i2sb': I2SBTrainer,
        'norm_controlnet': NormControlNetTrainer
    }

    return __TYPE_CLS_DICT[type](weight_dtype, accelerator, logger, cfg)
