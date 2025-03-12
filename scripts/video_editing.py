import torch
import os
from PIL import Image
from diffusers.utils import export_to_video
import cv2
import imageio
import argparse

from afldm.pipelines.video_equiv_editing_pipeline import VideoEquivariantEditingPipeline
from afldm.af_modules.af_api import make_af_unet, make_af_vae_from_config


def video_to_frame(video_path: str, interval: int, height: int = 512, width: int = 512):
    vidcap = cv2.VideoCapture(video_path)
    success = True

    count = 0
    res = []
    while success:
        count += 1
        success, image = vidcap.read()
        if count % interval != 1 and interval != 1:
            continue
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, [width, height])
            res.append(image)

    vidcap.release()
    return res


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        default='assets/car-turn.mp4')
    parser.add_argument('--output_path', type=str, default='results/car.gif')
    parser.add_argument('--prompt', type=str, default='a red car is turning')
    parser.add_argument('--inv_prompt', type=str, default='')
    parser.add_argument('--n_prompt', type=str, default='ugly, old, mutation, lowres, low quality, doll, long neck, extra limbs, text, signature, artist name, bad anatomy, poorly drawn, malformed, deformed, blurry, out of focus, noise, dust, crop')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--strength', type=float, default=0.7)
    parser.add_argument('--n_frames', type=int, default=None)
    parser.add_argument('-mp4', default=False, action='store_true')
    parser.add_argument('-no_af', default=False, action='store_true')
    return parser.parse_args()


H = 512
W = 512
if __name__ == '__main__':
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path
    prompt = args.prompt
    inv_prompt = args.inv_prompt
    n_prompt = args.n_prompt
    n_steps = args.n_steps
    strength = args.strength

    torch.manual_seed(0)

    frames = video_to_frame(input_path, 1, H, W)
    frames = [Image.fromarray(frame) for frame in frames]

    if args.n_frames is not None:
        frames = frames[:args.n_frames]

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    generator = torch.manual_seed(0)

    if args.no_af:
        pipe: VideoEquivariantEditingPipeline = VideoEquivariantEditingPipeline.from_pretrained(
            'stable-diffusion-v1-5/stable-diffusion-v1-5'
        ).to('cuda')
    else:
        pipe: VideoEquivariantEditingPipeline = VideoEquivariantEditingPipeline.from_pretrained(
            'SingleZombie/alias_free_sd15'
        ).to('cuda')

        make_af_unet(pipe.unet)
        make_af_vae_from_config(pipe.vae)

    frames = pipe(frames, prompt=prompt, negative_prompt=n_prompt, inv_prompt=inv_prompt,
                  guidance_scale=args.guidance_scale,
                  num_inference_steps=n_steps,
                  generator=generator, strength=strength).images

    if args.mp4:
        export_to_video(frames, output_path)
    else:
        imageio.mimsave(output_path, frames,
                        'GIF', duration=0.2, loop=0)
