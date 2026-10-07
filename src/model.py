import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    """
    Two convolution layers back to back. This is the basic repeated unit
    used at every stage (encoder, bottleneck, decoder) — it's what actually
    looks at a pixel and its neighbors and decides what matters.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
class SmallUNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        c = base_channels

        # Encoder: each stage looks at a coarser, more zoomed-out view
        self.enc1 = conv_block(1, c)
        self.enc2 = conv_block(c, c * 2)
        self.enc3 = conv_block(c * 2, c * 4)

        self.pool = nn.MaxPool2d(2)  # this is what actually shrinks the image by half

        # Bottleneck: the most zoomed-out point
        self.bottleneck = conv_block(c * 4, c * 8)

        # Decoder: each stage grows the image back up, combining with the
        # matching encoder stage's saved detail (the skip connection)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(c * 8, c * 4)  # c*8 because we concatenate with the skip connection

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = conv_block(c * 2, c)

        # Final layer: squash everything back down to 1 channel (grayscale output)
        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        # Going down (encoder), saving each stage's output for later
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck: most zoomed-out point
        b = self.bottleneck(self.pool(e3))

        # Going back up (decoder), combining with saved encoder outputs (skip connections)
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # torch.cat = "combine the two side by side"

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)