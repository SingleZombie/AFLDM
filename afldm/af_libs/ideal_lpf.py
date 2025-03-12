import torch
import torch.nn as nn


def create_lpf_rect(N, cutoff=0.5):
    cutoff_low = int((N * cutoff) // 2)
    cutoff_high = int(N - cutoff_low)
    rect_1d = torch.ones(N)
    rect_1d[cutoff_low + 1:cutoff_high] = 0
    if N % 4 == 0:
        # if N is divides by 4, nyquist freq should be 0
        # N % 4 =0 means the downsampeled signal is even
        rect_1d[cutoff_low] = 0
        rect_1d[cutoff_high] = 0

    rect_2d = rect_1d[:, None] * rect_1d[None, :]
    return rect_2d


def create_fixed_lpf_rect(N, size):
    rect_1d = torch.ones(N)
    if size < N:
        cutoff_low = size // 2
        cutoff_high = int(N - cutoff_low)
        rect_1d[cutoff_low + 1:cutoff_high] = 0
    rect_2d = rect_1d[:, None] * rect_1d[None, :]
    return rect_2d


# upsample using FFT
def create_recon_rect(N, cutoff=0.5):
    cutoff_low = int((N * cutoff) // 2)
    cutoff_high = int(N - cutoff_low)
    rect_1d = torch.ones(N)
    rect_1d[cutoff_low + 1:cutoff_high] = 0
    if N % 4 == 0:
        # if N is divides by 4, nyquist freq should be 0.5
        # N % 4 =0 means the downsampeled signal is even
        rect_1d[cutoff_low] = 0.5
        rect_1d[cutoff_high] = 0.5
    rect_2d = rect_1d[:, None] * rect_1d[None, :]
    return rect_2d


class LPF_RFFT(nn.Module):
    '''
    saves rect in first use
    '''

    def __init__(self, cutoff=0.5, transform_mode='rfft', fixed_size=None):
        super(LPF_RFFT, self).__init__()
        self.cutoff = cutoff
        self.fixed_size = fixed_size
        assert transform_mode in [
            'fft', 'rfft'], f'transform_mode={transform_mode} is not supported'
        self.rect_dict = {}
        self.transform_mode = transform_mode
        self.transform = torch.fft.fft2 if transform_mode == 'fft' else torch.fft.rfft2
        self.itransform = (lambda x: torch.real(torch.fft.ifft2(
            x))) if transform_mode == 'fft' else torch.fft.irfft2

    def forward(self, x):
        x_fft = self.transform(x)
        # if not hasattr(self, 'rect'):
        #     N = x.shape[-1]
        #     rect = create_lpf_rect(
        #         N, self.cutoff) if not self.fixed_size else create_fixed_lpf_rect(N, self.fixed_size)
        #     rect = rect[:, :int(
        #         N/2+1)] if self.transform_mode == 'rfft' else rect
        #     self.register_buffer('rect', rect, persistent=False)
        #     self.to(x.device)

        N = x.shape[-1]
        if N in self.rect_dict:
            rect = self.rect_dict[N]
        else:
            rect = create_lpf_rect(N, self.cutoff)
            rect = rect[:, :int(
                N/2+1)] if self.transform_mode == 'rfft' else rect
            rect = rect.to(x.device)
            self.rect_dict[N] = rect
        x_fft *= rect
        # out = self.itransform(x_fft) # support odd inputs - need to specify signal size (irfft default is even)
        out = self.itransform(x_fft, s=(x.shape[-2], x.shape[-1]))

        return out


class LPF_RECON_RFFT(nn.Module):
    '''
        saves rect in first use
        '''

    def __init__(self, cutoff=0.5, transform_mode='rfft'):
        super(LPF_RECON_RFFT, self).__init__()
        self.cutoff = cutoff
        assert transform_mode in [
            'fft', 'rfft'], f'mode={transform_mode} is not supported'
        self.rect_dict = {}
        self.transform_mode = transform_mode
        self.transform = torch.fft.fft2 if transform_mode == 'fft' else torch.fft.rfft2
        self.itransform = (lambda x: torch.real(torch.fft.ifft2(
            x))) if transform_mode == 'fft' else torch.fft.irfft2

    def forward(self, x):
        x_fft = self.transform(x)
        N = x.shape[-1]
        if N in self.rect_dict:
            rect = self.rect_dict[N].to(x.device)
        else:
            rect = create_recon_rect(N, self.cutoff)
            rect = rect[:, :int(N / 2 + 1)
                        ] if self.transform_mode == 'rfft' else rect
            rect = rect.to(x.device)
            self.rect_dict[N] = rect
            # self.register_buffer('rect', rect, persistent=False)
            # self.to(x.device)
        # if not hasattr(self, 'rect'):
        #     N = x.shape[-1]
        #     rect = create_recon_rect(N, self.cutoff)
        #     rect = rect[:, :int(N / 2 + 1)
        #                 ] if self.transform_mode == 'rfft' else rect
        #     self.register_buffer('rect', rect, persistent=False)
        #     self.to(x.device)
        x_fft *= rect
        out = self.itransform(x_fft)
        return out


class UpsampleRFFT(nn.Module):
    '''
    input shape is unknown
    '''

    def __init__(self, up=2, transform_mode='rfft', factor=1):
        super(UpsampleRFFT, self).__init__()
        self.up = up
        self.recon_filter = LPF_RECON_RFFT(
            cutoff=1 / up * factor, transform_mode=transform_mode)

    def forward(self, x):
        # pad zeros
        batch_size, num_channels, in_height, in_width = x.shape
        x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
        # x = x.repeat([1, 1, 1, self.up, 1, self.up])
        x = torch.nn.functional.pad(x, [0, self.up - 1, 0, 0, 0, self.up - 1])
        x = x.reshape([batch_size, num_channels, in_height *
                      self.up, in_width * self.up])
        x = self.recon_filter(x) * (self.up ** 2)
        # x = torch.roll(x, [1, 1], dims=[-2, -1])
        return x


def subpixel_shift(images, up=2, shift_x=1, shift_y=1, up_method='ideal'):
    '''
    effective fractional shift is (shift_x / up, shift_y / up)
    '''

    assert up_method == 'ideal', 'Only "ideal" interpolation kenrel is supported'
    up_layer = UpsampleRFFT(up=up).to(images.device)
    up_img_batch = up_layer(images)
    # img_batch_1 = up_img_batch[:, :, 1::2, 1::2]
    img_batch_1 = torch.roll(
        up_img_batch, shifts=(-shift_x, -shift_y), dims=(2, 3))[:, :, ::up, ::up]
    return img_batch_1
