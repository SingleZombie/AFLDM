
<h1 align="center"><strong>[CVPR 2025] Alias-free Latent Diffusion Models </strong></h1>
  <p align="center">
            <a href="https://zhouyifan.net/about/"> Yifan Zhou<sup>1</sup></a>
              <a href="https://github.com/xizaoqu">Zeqi Xiao<sup>1</sup></a	>
              <a href="https://williamyang1991.github.io/">Shuai Yang<sup>2</sup></a>
              <a href="https://xingangpan.github.io/">Xingang Pan<sup>1</sup></a>    <br>
    <sup>1</sup>S-Lab, Nanyang Technological University, <br> <sup>2</sup>Wangxuan Institute of Computer Technology, Peking University<br> 


<p align="center">
<a href="http://zhouyifan.net/AF-LDM-Page/">  Project Page </a> |
<a href="https://arxiv.org/abs/">  Paper </a>
</p>

https://github.com/user-attachments/assets/4fcd0c0f-4c0f-48a9-97dc-e5dcab9dd578

We found the VAE and denoising network in LDM are not equivariant to fractional shifts. We propose an alias-free framework to improve the fractional shift equivariance of LDM. We demonstrate the effectiveness of our method in various applications, including video editing, frame interpolation, super-resolution and normal estimation. 

## TODO

- [ ] Chinese/English blog posts
- [ ] Refine documents
- [ ] Training scripts

## Update

* \[03/2025\]: Repository created.

## Installation

1. Clone the repository. (Don't forget --recursive. Otherwise, please run git submodule update --init --recursive)

```shell
git clone git@github.com:SingleZombie/AFLDM.git --recursive
cd AFLDM
pip install -e .
```

2. Install PyTorch in your Python environment.

3. Install pip libraries.

```shell
pip install -r requirements.txt
```

## Inference

All the detailed commands are shown inside `.sh` files. 

### Unconditional Generation Shift

```shell
bash shift_ldm_ffhq.sh
```

### Video Editing

Due to the limitation of our computation resource, the finetuned alias-free Stable Diffusion has a poor generation capacity. It can only perform simple editing.

```shell
bash video_editing.sh
```

### Image Interpolation

```shell
bash image_interpolation.sh
```

### Super-resolution Shift

This is not a blind SR. The degradation function is fixed.

```shell
bash shift_ldm_sr.sh
```

### Normal Esitmation Shift

```shell
bash shift_normal_estimation.sh
```

## Citation

```
@inproceedings{zhou2025afldm,
      title={Alias-Free Latent Diffusion Models: Improving Fractional Shift Equivariance of Diffusion Latent Space},
      author={Zhou, Yifan and Xiao, Zeqi and Yang, Shuai and Pan, Xingang },
      booktitle = {CVPR},
      year = {2025},
    }
```

## Acknowledgements

* [Diffusers](https://github.com/huggingface/diffusers): Our project is bulit on diffusers.
* [GMFlow](https://github.com/haofeixu/gmflow): Our flow estimator.
* [StyleGAN3](https://github.com/NVlabs/stylegan3): For sharing alias-free module implementation. 
* [Alias-Free Convnets](https://github.com/hmichaeli/alias_free_convnets): For sharing alias-free module implementation. 
* [I2SB](https://github.com/NVlabs/I2SB): For sharing SR implementation.
* [StableNormal](https://github.com/Stable-X/StableNormal): For sharing normal estimation dataset.