# Demo 11: Deeper U-Net-style streaming model.
# Scales the encoder-decoder depth and channel count to demonstrate a larger receptive field.

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from tqdm import tqdm


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            stride=stride,
            bias=bias,
        )
        self.ring_buffer_length = (kernel_size - 1) * dilation

    def reset_state(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> None:
        self.ring_buffer = torch.zeros(batch_size, self.conv.in_channels, self.ring_buffer_length, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # implement "ring buffer"
        x_pad = torch.concatenate([self.ring_buffer, x], axis=2)
        # compute output on padded input
        y = self.conv(x_pad)
        # prepare ring buffer for next forward pass
        self.ring_buffer = x_pad[:, :, -self.ring_buffer_length:]
        return y


class ConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            stride=stride,
            bias=bias,
        )
        self.ring_buffer_length = (kernel_size - 1) * dilation + 1 - stride

    def reset_state(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> None:
        self.ring_buffer = torch.zeros(batch_size, self.conv.out_channels, self.ring_buffer_length, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward pass on input buffer
        y = self.conv(x)
        # accumulate output from previous forward pass
        y[:, :, :self.ring_buffer_length] += self.ring_buffer
        # prepare ring buffer for next forward pass
        self.ring_buffer = y[:, :, -self.ring_buffer_length:]
        # truncate ending which is still waiting for contributions from future forward passes
        # making the output the same length as the input
        return y[:, :, :-self.ring_buffer_length]


if __name__ == '__main__':
    batch_size = 1
    num_samples = 2**18
    kernel_size = 5
    in_channels = 1
    hidden_channels = 32
    out_channels = 1

    dtype = torch.float32
    device = 'cpu'

    x = torch.zeros(batch_size, in_channels, num_samples, dtype=dtype, device=device)
    # Use a centered impulse to expose each model's temporal response.
    x[:, :, num_samples // 2] = 1.0

    down0 = Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1)
    down1 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    down2 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    down3 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    down4 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    down5 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    down6 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    down7 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    down8 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)

    up0 = ConvTranspose1d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1)
    up1 = ConvTranspose1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    up2 = ConvTranspose1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    up3 = ConvTranspose1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    up4 = ConvTranspose1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    up5 = ConvTranspose1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    up6 = ConvTranspose1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    up7 = ConvTranspose1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    up8 = ConvTranspose1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)

    # reset states before processing
    down0.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    down1.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    down2.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    down3.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    down4.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    down5.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    down6.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    down7.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    down8.reset_state(batch_size=batch_size, dtype=dtype, device=device)

    up0.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    up1.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    up2.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    up3.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    up4.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    up5.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    up6.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    up7.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    up8.reset_state(batch_size=batch_size, dtype=dtype, device=device)

    # let's process 256 samples at a time - simulating low-latency inference
    # the minimum we can use is 2^8 = 256
    # Buffer size must be divisible by the total downsampling factor across 8 stride-4 stages.
    buffer_size = 4**8
    num_buffers = num_samples // buffer_size

    y0 = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    y6 = []
    y7 = []
    y8 = []

    z0 = []
    z1 = []
    z2 = []
    z3 = []
    z4 = []
    z5 = []
    z6 = []
    z7 = []
    z8 = []

    for i in tqdm(range(num_buffers)):
        x_i = x[:, :, i * buffer_size:(i + 1) * buffer_size]
        y0_i = down0(x_i)
        y1_i = down1(y0_i)
        y2_i = down2(y1_i)
        y3_i = down3(y2_i)
        y4_i = down4(y3_i)
        y5_i = down5(y4_i)
        y6_i = down6(y5_i)
        y7_i = down7(y6_i)
        y8_i = down8(y7_i)

        z8_i = up8(y8_i)
        z7_i = up7(z8_i)
        z6_i = up6(z7_i)
        z5_i = up5(z6_i)
        z4_i = up4(z5_i)
        z3_i = up3(z4_i)
        z2_i = up2(z3_i)
        z1_i = up1(z2_i)
        z0_i = up0(z1_i)

        y0 += [y0_i]
        y1 += [y1_i]
        y2 += [y2_i]
        y3 += [y3_i]
        y4 += [y4_i]
        y5 += [y5_i]
        y6 += [y6_i]
        y7 += [y7_i]
        y8 += [y8_i]

        z0 += [z0_i]
        z1 += [z1_i]
        z2 += [z2_i]
        z3 += [z3_i]
        z4 += [z4_i]
        z5 += [z5_i]
        z6 += [z6_i]
        z7 += [z7_i]
        z8 += [z8_i]

    # concatenate outputs to create final output
    y0 = torch.concatenate(y0, dim=2)
    y1 = torch.concatenate(y1, dim=2)
    y2 = torch.concatenate(y2, dim=2)
    y3 = torch.concatenate(y3, dim=2)
    y4 = torch.concatenate(y4, dim=2)
    y5 = torch.concatenate(y5, dim=2)
    y6 = torch.concatenate(y6, dim=2)
    y7 = torch.concatenate(y7, dim=2)
    y8 = torch.concatenate(y8, dim=2)

    z0 = torch.concatenate(z0, dim=2)
    z1 = torch.concatenate(z1, dim=2)
    z2 = torch.concatenate(z2, dim=2)
    z3 = torch.concatenate(z3, dim=2)
    z4 = torch.concatenate(z4, dim=2)
    z5 = torch.concatenate(z5, dim=2)
    z6 = torch.concatenate(z6, dim=2)
    z7 = torch.concatenate(z7, dim=2)
    z8 = torch.concatenate(z8, dim=2)

    import matplotlib.pyplot as plt
    x = x.squeeze().cpu().numpy()
    y0 = y0.squeeze().detach().cpu().numpy()[0]
    y1 = y1.squeeze().detach().cpu().numpy()[0]
    y2 = y2.squeeze().detach().cpu().numpy()[0]
    y3 = y3.squeeze().detach().cpu().numpy()[0]
    y4 = y4.squeeze().detach().cpu().numpy()[0]
    y5 = y5.squeeze().detach().cpu().numpy()[0]
    y6 = y6.squeeze().detach().cpu().numpy()[0]
    y7 = y7.squeeze().detach().cpu().numpy()[0]
    y8 = y8.squeeze().detach().cpu().numpy()[0]

    z0 = z0.squeeze().detach().cpu().numpy()
    z1 = z1.squeeze().detach().cpu().numpy()[0]
    z2 = z2.squeeze().detach().cpu().numpy()[0]
    z3 = z3.squeeze().detach().cpu().numpy()[0]
    z4 = z4.squeeze().detach().cpu().numpy()[0]
    z5 = z5.squeeze().detach().cpu().numpy()[0]
    z6 = z6.squeeze().detach().cpu().numpy()[0]
    z7 = z7.squeeze().detach().cpu().numpy()[0]
    z8 = z8.squeeze().detach().cpu().numpy()[0]

    fig, axes = plt.subplots(10, 2, figsize=(10, 15), sharex=True)
    axes[0, 0].plot(x)
    axes[0, 1].plot(z0)

    axes[1, 0].plot(y0)
    axes[1, 1].plot(z1)

    axes[2, 0].plot(y1)
    axes[2, 1].plot(z2)

    axes[3, 0].plot(y2)
    axes[3, 1].plot(z3)

    axes[4, 0].plot(y3)
    axes[4, 1].plot(z4)

    axes[5, 0].plot(y4)
    axes[5, 1].plot(z5)

    axes[6, 0].plot(y5)
    axes[6, 1].plot(z6)

    axes[7, 0].plot(y6)
    axes[7, 1].plot(z7)

    axes[8, 0].plot(y7)
    axes[8, 1].plot(z8)

    axes[9, 0].plot(y8)
    axes[9, 1].plot(y8)

    for i, ax in enumerate(axes.flatten()):
        ax.grid()
        ax.set_xlabel('Time (samples)')

    for i in range(10):
        if i == 0:
            axes[i, 0].set_title('Input')
            axes[i, 1].set_title('Output')
        else:
            axes[i, 0].set_title(f'Down {i-1}')
            axes[i, 1].set_title(f'Up {i-1}')

    sf.write('output/11_deeper_impulse_response.wav', z0 / np.max(np.abs(z0)), 16000)

    plt.tight_layout()
    plt.show()
