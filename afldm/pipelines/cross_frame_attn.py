
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0


class AttnState:
    STORE = 0
    LOAD = 1
    IDLE = 2

    def __init__(self):
        self.reset()

    @property
    def state(self):
        return self.__state

    @property
    def alpha(self):
        return self.__alpha

    @property
    def store_id(self):
        return self.__store_id

    @property
    def timestep(self):
        return self.__timestep

    def set_timestep(self, t):
        if isinstance(t, torch.Tensor):
            t = t.item()
        self.__timestep = t

    def set_alpha(self, alpha):
        self.__alpha = alpha

    def set_store_id(self, store_id):
        self.__store_id = store_id

    def reset(self):
        self.__state = AttnState.STORE
        self.__timestep = 0
        self.__store_id = 0
        self.__alpha = 0

    def to_load(self):
        self.__state = AttnState.LOAD

    def to_idle(self):
        self.__state = AttnState.IDLE


class CrossFrameAttnProcessor(AttnProcessor2_0):
    """
    Args:
        attn_state: Whether the model is processing the first frame or an intermediate frame
    """

    def __init__(self, attn_state: AttnState, enable_interp=False):
        super().__init__()
        self.attn_state = attn_state
        self.maps = [dict(), dict()]
        self.enable_interp = enable_interp

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        # Is self attention
        if encoder_hidden_states is None:
            t = self.attn_state.timestep
            if self.attn_state.state == AttnState.IDLE:
                res = super().__call__(attn, hidden_states,
                                       encoder_hidden_states, attention_mask, temb)
            elif self.attn_state.state == AttnState.STORE:
                self.maps[self.attn_state.store_id][t] = hidden_states.detach()
                res = super().__call__(attn, hidden_states,
                                       encoder_hidden_states, attention_mask, temb)
            elif self.attn_state.state == AttnState.LOAD:

                if self.maps[0][t].ndim == 4:
                    n, c, h, w = self.maps[0][t].shape
                    reshaped_map = self.maps[0][t].view(
                        n, c, h * w).transpose(1, 2)
                else:
                    reshaped_map = self.maps[0][t]

                if attn.group_norm is not None:
                    reshaped_map = attn.group_norm(
                        reshaped_map.transpose(1, 2)).transpose(1, 2)

                # Match the batch size of the hidden states
                if reshaped_map.shape[0] < hidden_states.shape[0]:
                    n, c, x = reshaped_map.shape
                    reshaped_map = reshaped_map.unsqueeze(1)
                    map0 = reshaped_map.repeat(
                        1, hidden_states.shape[0] // reshaped_map.shape[0], 1, 1).reshape(hidden_states.shape[0], c, x)
                else:
                    map0 = reshaped_map

                # For morphing/interpolation
                if self.enable_interp:
                    alpha = self.attn_state.alpha

                    if self.maps[1][t].ndim == 4:
                        n, c, h, w = self.maps[1][t].shape
                        reshaped_map = self.maps[1][t].view(
                            n, c, h * w).transpose(1, 2)
                    else:
                        reshaped_map = self.maps[1][t]
                    if attn.group_norm is not None:
                        reshaped_map = attn.group_norm(
                            reshaped_map.transpose(1, 2)).transpose(1, 2)
                    if reshaped_map.shape[0] < hidden_states.shape[0]:
                        n, c, x = reshaped_map.shape
                        reshaped_map = reshaped_map.unsqueeze(1)
                        map1 = reshaped_map.repeat(
                            1, hidden_states.shape[0] // reshaped_map.shape[0], 1, 1).reshape(hidden_states.shape[0], c, x)
                    else:
                        map1 = reshaped_map

                    res1 = super().__call__(attn, hidden_states, map0, attention_mask, temb)
                    res2 = super().__call__(attn, hidden_states, map1, attention_mask, temb)
                    res = (1 - alpha) * res1 + alpha * res2
                # Standard Cross-Frame Attention
                else:
                    res = super().__call__(attn, hidden_states, map0, attention_mask, temb)
        else:
            res = super().__call__(attn, hidden_states,
                                   encoder_hidden_states, attention_mask, temb)

        return res


def get_unet_attn_processors(unet):
    r"""
    Returns:
        `dict` of attention processors: A dictionary containing all attention processors used in the model with
        indexed by its weight name.
    """
    # set recursively
    processors = {}

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
        if hasattr(module, "get_processor"):
            processors[f"{name}.processor"] = module.get_processor()

        for sub_name, child in module.named_children():
            fn_recursive_add_processors(
                f"{name}.{sub_name}", child, processors)

        return processors

    for name, module in unet.named_children():
        fn_recursive_add_processors(name, module, processors)

    return processors


def set_unet_attn_processor(unet, processor):
    r"""
    Sets the attention processor to use to compute attention.

    Parameters:
        processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
            The instantiated processor class or a dictionary of processor classes that will be set as the processor
            for **all** `Attention` layers.

            If `processor` is a dict, the key needs to define the path to the corresponding cross attention
            processor. This is strongly recommended when setting trainable attention processors.

    """
    count = len(get_unet_attn_processors(unet).keys())

    if isinstance(processor, dict) and len(processor) != count:
        raise ValueError(
            f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
            f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
        )

    def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
        if hasattr(module, "set_processor"):
            if not isinstance(processor, dict):
                module.set_processor(processor)
            else:
                module.set_processor(processor.pop(f"{name}.processor"))

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

    for name, module in unet.named_children():
        fn_recursive_attn_processor(name, module, processor)
