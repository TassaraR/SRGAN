import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    k3n64s1
    Conv -> Bn -> PReLU -> Conv -> Bn -> Element-wise sum
    """
    def __init__(self, in_channels=64, out_channels=64):
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_block = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels,
                                                 kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(self.out_channels),
                                       nn.PReLU(),
                                       nn.Conv2d(self.out_channels, self.out_channels,
                                                 kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(self.out_channels))

    def forward(self, x):
        block = self.res_block(x)
        return torch.add(x, block)


class UpSampleBlock(nn.Module):
    """
    k3n256s1
    Conv -> PixelShuffle -> PReLU
    """
    def __init__(self, in_channels=256, out_channels=256):
        super(UpSampleBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_block = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels,
                                                      kernel_size=3, stride=1, padding=1),
                                            nn.PixelShuffle(upscale_factor=2),
                                            nn.PReLU())

    def forward(self, x):
        return self.upsample_block(x)


class Generator(nn.Module):

    def __init__(self, im_chan=3, n_residual_blocks=16, n_upsample_blocks=2):
        super(Generator, self).__init__()

        self.im_chan = im_chan
        self.n_residual_blocks = n_residual_blocks
        self.n_upsample_blocks = n_upsample_blocks

        # k9n64s1
        self.conv1 = nn.Sequential(nn.Conv2d(self.im_chan, 64,
                                             kernel_size=9, stride=1, padding=4),
                                   nn.PReLU())
        self.residual_blocks = self.make_residual_blocks()
        # k2n64s1
        self.conv2 = nn.Sequential(nn.Conv2d(self.im_chan, 64,
                                             kernel_size=9, stride=1, padding=1),
                                   nn.PReLU())
        self.upsample_blocks = self.make_upsample_blocks()
        # k9n3s1
        self.conv3 = nn.Conv2d(64, self.im_chan,
                               kernel_size=9, stride=1, padding=4)

    def make_residual_blocks(self):
        res_blocks = []
        for _ in range(self.n_residual_blocks):
            res_blocks.append(ResidualBlock())
        return nn.Sequential(*res_blocks)

    def make_upsample_blocks(self):
        up_blocks = []
        for _ in range(self.n_upsample_blocks):
            up_blocks.append(UpSampleBlock())
        return nn.Sequential(*up_blocks)

    def forward(self, x):

        cv1_out = self.conv1(x)
        res_out = self.residual_blocks(cv1_out)
        cv2_out = self.conv2(res_out)
        elemwise_out = torch.add(cv2_out, cv1_out)
        up_out = self.upsample_blocks(elemwise_out)
        cv3_out = self.conv3(up_out)

        return cv3_out
