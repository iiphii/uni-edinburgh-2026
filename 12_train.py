# Demo 12: Training the stateful U-Net-style model on real audio pairs.
# Uses segment-wise streaming training so inference-time state handling is preserved during optimization.

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


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

    def detach_state(self):
        self.ring_buffer = self.ring_buffer.detach()

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

    def detach_state(self):
        self.ring_buffer = self.ring_buffer.detach()

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
    # Slow: Use GPU if available
    batch_size = 16
    segment_length = 2**17
    kernel_size = 5
    in_channels = 1
    hidden_channels = 32
    out_channels = 1

    dtype = torch.float32
    device = 'cpu'

    # load data
    x, fs = sf.read('data/input-16kHz.wav')
    y, fs = sf.read('data/target-16kHz.wav')

    x = torch.from_numpy(x).to(dtype=dtype, device=device)
    y = torch.from_numpy(y).to(dtype=dtype, device=device)

    # number of available training segments
    num_segments = x.shape[0] // segment_length
    segment_offset = num_segments // batch_size
    num_samples = num_segments * segment_length

    x = x[:num_samples]
    y = y[:num_samples]

    down0 = Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1)
    down1 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    down2 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    down3 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    down4 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    down5 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    down6 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    down7 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    down8 = Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)

    up0 = ConvTranspose1d(in_channels=2 * hidden_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1)
    up1 = ConvTranspose1d(in_channels=2 * hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    up2 = ConvTranspose1d(in_channels=2 * hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    up3 = ConvTranspose1d(in_channels=2 * hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    up4 = ConvTranspose1d(in_channels=2 * hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    up5 = ConvTranspose1d(in_channels=2 * hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    up6 = ConvTranspose1d(in_channels=2 * hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    up7 = ConvTranspose1d(in_channels=2 * hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)
    up8 = ConvTranspose1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=4)

    layers = nn.ModuleList([
        down0,
        down1,
        down2,
        down3,
        down4,
        down5,
        down6,
        down7,
        down8,
        up0,
        up1,
        up2,
        up3,
        up4,
        up5,
        up6,
        up7,
        up8,
    ])

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

    optimizer = torch.optim.Adam(params=layers.parameters())

    num_iterations = 10000
    # Each iteration trains on contiguous wrapped segments while preserving recurrent state.
    for i in range(num_iterations):
        # construct training batch
        X = []
        Y = []
        for j in range(batch_size):
            # starting index for this batch element at this segment
            start = (j * segment_offset + i * segment_length) % num_samples

            # indices for this segment (wrap around if needed)
            idx = (start + np.arange(segment_length)) % num_samples

            X += x[idx]
            Y += y[idx]

        X = torch.stack(X, dim=0)
        Y = torch.stack(Y, dim=0)

        # add extra channel dimension
        X = X.reshape(batch_size, 1, segment_length)
        Y = Y.reshape(batch_size, 1, segment_length)

        # model forward pass - finally adding nonlinearities
        Y0 = torch.tanh(down0(X))
        Y1 = torch.tanh(down1(Y0))
        Y2 = torch.tanh(down2(Y1))
        Y3 = torch.tanh(down3(Y2))
        Y4 = torch.tanh(down4(Y3))
        Y5 = torch.tanh(down5(Y4))
        Y6 = torch.tanh(down6(Y5))
        Y7 = torch.tanh(down7(Y6))
        Y8 = torch.tanh(down8(Y7))

        Z8 = torch.tanh(up8(Y8))
        Z7 = torch.tanh(up7(torch.concatenate([Y7, Z8], dim=1)))
        Z6 = torch.tanh(up6(torch.concatenate([Y6, Z7], dim=1)))
        Z5 = torch.tanh(up5(torch.concatenate([Y5, Z6], dim=1)))
        Z4 = torch.tanh(up4(torch.concatenate([Y4, Z5], dim=1)))
        Z3 = torch.tanh(up3(torch.concatenate([Y3, Z4], dim=1)))
        Z2 = torch.tanh(up2(torch.concatenate([Y2, Z3], dim=1)))
        Z1 = torch.tanh(up1(torch.concatenate([Y1, Z2], dim=1)))
        Z0 = torch.tanh(up0(torch.concatenate([Y0, Z1], dim=1)))

        # mean square error
        loss = (Y - Z0).square().mean()

        # Backprop through this iteration, then detach recurrent buffers to truncate history.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        down0.detach_state()
        down1.detach_state()
        down2.detach_state()
        down3.detach_state()
        down4.detach_state()
        down5.detach_state()
        down6.detach_state()
        down7.detach_state()
        down8.detach_state()

        up0.detach_state()
        up1.detach_state()
        up2.detach_state()
        up3.detach_state()
        up4.detach_state()
        up5.detach_state()
        up6.detach_state()
        up7.detach_state()
        up8.detach_state()

        print(f'Iteration {i}, loss: {loss.item()}')
