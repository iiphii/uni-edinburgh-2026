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
    layer2 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
    layer3 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
    layer4 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    y0 = layer0(x)
    y1 = layer1(y0)
    y2 = layer2(y1)
    y3 = layer3(y2)
    y4 = layer4(y3)

    import matplotlib.pyplot as plt
    x = x.squeeze().cpu().numpy()
    y0 = y1.squeeze().detach().cpu().numpy()
    y1 = y1.squeeze().detach().cpu().numpy()
    y2 = y2.squeeze().detach().cpu().numpy()
    y3 = y3.squeeze().detach().cpu().numpy()
    y4 = y4.squeeze().detach().cpu().numpy()

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
    ax1.plot(x)
    ax2.plot(y0)
    ax3.plot(y1)
    ax4.plot(y2)
    ax5.plot(y3)
    ax6.plot(y4)
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
