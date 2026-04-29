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


if __name__ == '__main__':
    batch_size = 1
    num_samples = 256
    kernel_size = 3
    in_channels = 1
    out_channels = 1

    dtype = torch.float32
    device = 'cpu'

    x = torch.zeros(batch_size, in_channels, num_samples, dtype=dtype, device=device)
    x[:, :, num_samples // 2] = 1.0

    layer0 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=1)
    layer1 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=2)
    layer2 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=4)
    layer3 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=8)
    layer4 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=16)

    # reset states before processing
    layer0.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    layer1.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    layer2.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    layer3.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    layer4.reset_state(batch_size=batch_size, dtype=dtype, device=device)

    # let's process 4 samples at a time - simulating low-latency inference
    buffer_size = 4
    num_buffers = num_samples // buffer_size

    y0 = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []

    for i in tqdm(range(num_buffers)):
        x_i = x[:, :, i * buffer_size:(i + 1) * buffer_size]
        y0_i = layer0(x_i)
        y1_i = layer1(y0_i)
        y2_i = layer2(y1_i)
        y3_i = layer3(y2_i)
        y4_i = layer4(y3_i)

        y0 += [y0_i]
        y1 += [y1_i]
        y2 += [y2_i]
        y3 += [y3_i]
        y4 += [y4_i]

    # concatenate outputs to create final output
    y0 = torch.concatenate(y0, dim=2)
    y1 = torch.concatenate(y1, dim=2)
    y2 = torch.concatenate(y2, dim=2)
    y3 = torch.concatenate(y3, dim=2)
    y4 = torch.concatenate(y4, dim=2)

    # let's compare against full signal processed in one go
    layer0.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    layer1.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    layer2.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    layer3.reset_state(batch_size=batch_size, dtype=dtype, device=device)
    layer4.reset_state(batch_size=batch_size, dtype=dtype, device=device)

    y0_full = layer0(x)
    y1_full = layer1(y0_full)
    y2_full = layer2(y1_full)
    y3_full = layer3(y2_full)
    y4_full = layer4(y3_full)

    import matplotlib.pyplot as plt
    x = x.squeeze().cpu().numpy()
    y0 = y0.squeeze().detach().cpu().numpy()
    y1 = y1.squeeze().detach().cpu().numpy()
    y2 = y2.squeeze().detach().cpu().numpy()
    y3 = y3.squeeze().detach().cpu().numpy()
    y4 = y4.squeeze().detach().cpu().numpy()

    y0_full = y0_full.squeeze().detach().cpu().numpy()
    y1_full = y1_full.squeeze().detach().cpu().numpy()
    y2_full = y2_full.squeeze().detach().cpu().numpy()
    y3_full = y3_full.squeeze().detach().cpu().numpy()
    y4_full = y4_full.squeeze().detach().cpu().numpy()

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
    ax1.plot(x)

    ax2.plot(y0)
    ax2.plot(y0_full, linestyle='--')

    ax3.plot(y1)
    ax3.plot(y1_full, linestyle='--')

    ax4.plot(y2)
    ax4.plot(y2_full, linestyle='--')

    ax5.plot(y3)
    ax5.plot(y3_full, linestyle='--')

    ax6.plot(y4)
    ax6.plot(y4_full, linestyle='--')

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax5.grid()
    ax6.grid()
    ax1.set_xlabel('Time (samples)')
    ax2.set_xlabel('Time (samples)')
    ax3.set_xlabel('Time (samples)')
    ax4.set_xlabel('Time (samples)')
    ax5.set_xlabel('Time (samples)')
    ax6.set_xlabel('Time (samples)')
    ax1.set_title('Input')
    ax2.set_title('Layer 0')
    ax3.set_title('Layer 1')
    ax4.set_title('Layer 2')
    ax5.set_title('Layer 3')
    ax6.set_title('Layer 4')

    plt.tight_layout()
    plt.show()
