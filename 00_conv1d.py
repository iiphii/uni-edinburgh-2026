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

    y = layer0(x)

    import matplotlib.pyplot as plt
    x = x.squeeze().cpu().numpy()
    y = y.squeeze().detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(x)
    ax2.plot(y)
    ax1.grid()
    ax2.grid()
    ax1.set_xlabel('Time (samples)')
    ax2.set_xlabel('Time (samples)')
    ax1.set_title('Input')
    ax2.set_title('Output')

    plt.tight_layout()
    plt.show()
