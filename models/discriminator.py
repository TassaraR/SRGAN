import torch
from torch import nn


class Discriminator(nn.Module):

    def __init__(self, in_channels=3, out_channels=64, n_conv_blocks=7):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_conv_blocks = n_conv_blocks

        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels,
                                             kernel_size=1, stride=1, padding=1),
                                   nn.LeakyReLU(0.2))

        self.conv_sequence = self.make_conv_blocks()

        # Defaults to 512
        in_class = self.output * 2**(self.n_conv_blocks//2)
        # Defaults to 1024
        out_class = in_class * 2
        self.classifier_block = nn.Sequential(nn.Linear(in_class, out_class),
                                              nn.LeakyReLU(0.2),
                                              nn.Linear(out_class, 1),
                                              nn.Sigmoid())

    def make_conv_blocks(self):
        blocks = []
        out_chan = self.out_channels
        for i in range(self.n_conv_blocks):
            stride = 1 if (i + 1) % 2 == 0 else 2
            if i == 0:
                conv = nn.Conv2d(self.out_channels, out_chan,
                                 kernel_size=3, stride=stride, padding=1)
            elif (i+1) % 2 == 0:
                out_chan = out_chan * 2
                conv = nn.Conv2d(out_chan // 2, out_chan,
                                 kernel_size=3, stride=stride, padding=1)
            elif (i+1) % 2 != 0:
                conv = nn.Conv2d(out_chan, out_chan,
                                 kernel_size=3, stride=stride, padding=1)
            blocks.append(nn.Sequential(conv,
                                        nn.BatchNorm2d(out_chan),
                                        nn.LeakyReLU(0.2)))
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv_sequence(out)
        out = self.classifier_block(out)
        return out
