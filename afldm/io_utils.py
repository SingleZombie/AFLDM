import os

import imageio
import torch
from diffusers.utils import load_image
from einops import rearrange
from torchvision import transforms


def image_to_tensor(files, resolution=(512, 512)):
    if not isinstance(files, list):
        files = [files]

    if resolution is None:
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    else:
        train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    tensor_list = []

    for file in files:
        img = load_image(file)

        tensor = train_transforms(img).unsqueeze(0)
        tensor_list.append(tensor)

    return torch.cat(tensor_list)


def save_gif_from_tensors(tensors, output_gif_path, duration=0.5, denorm=False):

    pil_imgs = []
    for tensor in tensors:
        if denorm:
            tensor = (tensor + 1) / 2
        if tensor.ndim == 4:
            tensor = rearrange(tensor, 'n c h w -> c h (n w)')
        if tensor.shape[1] == 4:
            tensor = tensor[:, :3]
        tensor = torch.clamp(tensor, 0, 1)
        pil_img = transforms.ToPILImage()(tensor)
        pil_imgs.append(pil_img)

    dir, _ = os.path.split(output_gif_path)
    os.makedirs(dir, exist_ok=True)

    imageio.mimsave(output_gif_path, pil_imgs,
                    'GIF', duration=duration, loop=0)
