

import torch
import os
from PIL import Image
import imageio
import requests
import argparse

from afldm.pipelines.image_interpolation_pipeline import ImageInterpolationPipeline
from afldm.af_modules.af_api import make_af_unet, make_af_vae_from_config


def download(url, dir, name=None):
    os.makedirs(dir, exist_ok=True)
    if name is None:
        name = url.split('/')[-1]
    path = os.path.join(dir, name)
    if not os.path.exists(path):
        print(f'Install {name} ...')
        open(path, 'wb').write(requests.get(url).content)
        print('Install successfully.')


def download_gmflow_ckpt():
    if not os.path.exists('gmflow'):
        raise FileNotFoundError(
            'Please install submodules `gmflow` with `git submodule update --init --recursive`')

    url = ('https://huggingface.co/PKUWilliamYang/Rerender/'
           'resolve/main/models/gmflow_sintel-0c07dcb3.pth')
    download(url, 'gmflow')


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_1', type=str,
                        default='assets/sleeping0.png')
    parser.add_argument('--input_path_2', type=str,
                        default='assets/sleeping1.png')
    parser.add_argument('--output_path', type=str,
                        default='results/sleeping_interpolation.gif')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--inv_prompt', type=str, default='')
    parser.add_argument('--n_prompt', type=str, default='')
    parser.add_argument('--n_frames', type=int, default=17)
    parser.add_argument('--n_steps', type=int, default=50)
    return parser.parse_args()


if __name__ == '__main__':
    download_gmflow_ckpt()

    pipe: ImageInterpolationPipeline = ImageInterpolationPipeline.from_pretrained(
        'SingleZombie/alias_free_sd15',
        # 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    ).to('cuda')

    make_af_unet(pipe.unet)
    make_af_vae_from_config(pipe.vae)

    args = parse_args()
    output_path = args.output_path
    path1 = args.input_path_1
    path2 = args.input_path_2
    n_steps = args.n_steps
    prompt = args.prompt
    n_prompt = args.n_prompt
    inv_prompt = args.inv_prompt

    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    generator = torch.manual_seed(0)

    frames = pipe(img1, img2, prompt=prompt,
                  negative_prompt=n_prompt,
                  inv_prompt=inv_prompt,
                  num_frames=args.n_frames,
                  num_inference_steps=n_steps,
                  warp_method=0,
                  enable_morph=True,
                  generator=generator).images

    imageio.mimsave(output_path, frames,
                    'GIF', duration=0.1, loop=0)
