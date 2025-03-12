import torch


def mask_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor):
    batch_loss = (a * mask - b * mask).square().sum((1, 2, 3)) / \
        mask.sum((1, 2, 3))
    return batch_loss.mean()


def mask_psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor):
    a_ = a * mask
    b_ = b * mask
    i_max = torch.max(a_.max(), b_.max()) - torch.min(a_.min(), b_.min())
    return 10 * torch.log10(i_max * i_max / mask_mse(a, b, mask))
