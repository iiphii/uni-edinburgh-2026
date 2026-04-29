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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


if __name__ == '__main__':
    batch_size = 1
    num_samples = 16
    kernel_size = 3
    in_channels = 1
    out_channels = 1

    dtype = torch.float32
    device = 'cpu'

    x = torch.zeros(batch_size, in_channels, num_samples, dtype=dtype, device=device)
    x[:, :, num_samples // 2] = 1.0

    layer0 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
    layer1 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    y0 = layer0(x)
    y1 = layer1(y0)

    import matplotlib.pyplot as plt
    x = x.squeeze().cpu().numpy()
    y0 = y0.squeeze().detach().cpu().numpy()
    y1 = y1.squeeze().detach().cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax1.plot(x)
    ax2.plot(y0)
    ax3.plot(y1)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.set_xlabel('Time (samples)')
    ax2.set_xlabel('Time (samples)')
    ax3.set_xlabel('Time (samples)')
    ax1.set_title('Input')
    ax2.set_title('Layer 0')
    ax3.set_title('Layer 1')

    plt.tight_layout()
    plt.show()
