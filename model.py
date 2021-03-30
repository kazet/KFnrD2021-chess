import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ConvResBlock, self).__init__()
        self.hidden_size = max(in_channels, out_channels)
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.hidden_size,
                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.GELU(), nn.BatchNorm2d(self.hidden_size))
        self.alpha = 0.
        self.active_layers = 0
        self.hidden = nn.Sequential(
            *(nn.Sequential(
                        nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size,
                                  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                        nn.GELU(), nn.BatchNorm2d(self.hidden_size)) for _ in range(16))
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_size, out_channels=out_channels,
                      kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1)),
            nn.GELU(), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        if self.alpha > 1.:
            self.alpha = 0.
            self.active_layers += 1
        h_out = [self.conv_in(x)]
        for h_layer in self.hidden[:self.active_layers]:
            h_out.append(h_layer(h_out[-1]) + h_out[-1])
        out_input = h_out[-1] + self.hidden[self.active_layers](h_out[-1]) * self.alpha
        out = self.conv_out(out_input)
        return out


class Coder(nn.Module):
    def __init__(self, input_shape, latent_size):
        super(Coder, self).__init__()
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.conv_in = ConvResBlock(in_channels=input_shape[0], out_channels=64)
        self.conv_mid = ConvResBlock(in_channels=64, out_channels=128)
        self.conv_out = ConvResBlock(in_channels=128, out_channels=256)
        self.nn = nn.Sequential(
            self.conv_in, nn.Dropout(.4),
            self.conv_mid, nn.Dropout(.2),
            self.conv_out,
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=max(128, latent_size * 2)), nn.GELU(),
            nn.Linear(in_features=max(128, latent_size * 2), out_features=latent_size))

    def alpha_ascent(self, increment_value):
        self.conv_in.alpha += increment_value
        self.conv_mid.alpha += increment_value
        self.conv_out.alpha += increment_value

    def forward(self, x):
        return self.nn(x)


class Decoder(nn.Module):
    def __init__(self, output_shape, latent_size):
        super(Decoder, self).__init__()
        self.output_shape = output_shape
        self.latent_size = latent_size
        self.conv_in = ConvResBlock(in_channels=128, out_channels=64, stride=1)
        self.conv_mid = ConvResBlock(in_channels=64, out_channels=64, stride=1)
        self.conv_out = ConvResBlock(in_channels=64, out_channels=64, stride=1)
        self.nn = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=max(256, latent_size * 2)), nn.GELU(),
            nn.Linear(in_features=max(256, latent_size * 2), out_features=128), nn.GELU(),
            nn.Unflatten(1, (128, 1, 1)),
            nn.UpsamplingNearest2d((2, 2)),
            self.conv_in,
            nn.UpsamplingNearest2d((4, 4)),
            self.conv_mid,
            nn.UpsamplingNearest2d((8, 8)),
            self.conv_out,
            nn.Conv2d(in_channels=64, out_channels=output_shape[0],
                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Sigmoid())

    def alpha_ascent(self, increment_value):
        self.conv_in.alpha += increment_value
        self.conv_mid.alpha += increment_value
        self.conv_out.alpha += increment_value

    def forward(self, x):
        return self.nn(x)


class Autoencoder(nn.Module):
    def __init__(self, input_shape, data_size):
        super(Autoencoder, self).__init__()
        self.coder = Coder(input_shape, data_size)
        self.decoder = Decoder(input_shape, data_size)

    def alpha_ascent(self, increment_value):
        self.coder.alpha_ascent(increment_value)
        self.decoder.alpha_ascent(increment_value)

    def forward(self, x):
        coded = self.coder(x)
        reconstructed = self.decoder(coded)
        return coded, reconstructed
