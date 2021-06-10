import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, noise_level=.017):
        super(NoisyLinear, self).__init__(in_features, out_features)
        self.weight_noise = nn.Parameter(torch.full((out_features, in_features), noise_level))
        self.bias_noise = nn.Parameter(torch.full((out_features,), noise_level))
        self.register_buffer("w_eps", torch.zeros(out_features, in_features))
        self.register_buffer("b_eps", torch.zeros(out_features,))

    def forward(self, x):
        w_noise = self.w_eps.data.normal_()
        b_noise = self.b_eps.data.normal_()
        w = self.weight + w_noise*self.weight_noise
        b = self.bias + b_noise*self.bias_noise
        return F.linear(x, w, b)


class NoiseGate(nn.Module):
    def __init__(self, n_features, noise_level=.017):
        super(NoiseGate, self).__init__()
        self.noise = nn.Parameter(torch.full((n_features,), noise_level))
        self.register_buffer("n_eps", torch.zeros(n_features))

    def forward(self, x):
        self.n_eps.data.normal_()
        noise = self.noise * self.n_eps
        return x + noise


class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ConvResBlock, self).__init__()
        self.hidden_size = max(in_channels, out_channels)
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.hidden_size,
                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.GELU(), nn.BatchNorm2d(self.hidden_size))
        self.hidden = nn.Sequential(
                nn.Conv2d(in_channels=self.hidden_size+in_channels, out_channels=self.hidden_size,
                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.GELU(), nn.BatchNorm2d(self.hidden_size)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_size*2, out_channels=out_channels,
                      kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1)),
            nn.GELU(), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        in_out = self.conv_in(x)
        hid_out = self.hidden(torch.cat([x, in_out], dim=1))
        out_out = self.conv_out(torch.cat([in_out, hid_out], dim=1))
        return out_out


class Coder(nn.Module):
    def __init__(self, input_shape, latent_size):
        super(Coder, self).__init__()
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.conv_in = ConvResBlock(in_channels=input_shape[0], out_channels=64)
        self.conv_mid = ConvResBlock(in_channels=64, out_channels=64)
        self.conv_out = ConvResBlock(in_channels=64, out_channels=128)
        self.nn = nn.Sequential(
            self.conv_in, nn.Dropout(.4),
            self.conv_mid, nn.Dropout(.2),
            self.conv_out,
            nn.Flatten(),
            NoisyLinear(in_features=128, out_features=max(128, latent_size * 2)), nn.GELU(),
            NoisyLinear(in_features=max(128, latent_size * 2), out_features=latent_size),
            NoiseGate(latent_size), nn.InstanceNorm1d(latent_size), nn.Dropout(.017))

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
            NoisyLinear(in_features=latent_size, out_features=max(128, latent_size * 2)), nn.GELU(),
            NoisyLinear(in_features=max(128, latent_size * 2), out_features=128), nn.GELU(),
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

    def forward(self, x):
        return self.nn(x)


class Autoencoder(nn.Module):
    def __init__(self, input_shape, data_size):
        super(Autoencoder, self).__init__()
        self.coder = Coder(input_shape, data_size)
        self.decoder = Decoder(input_shape, data_size)

    def forward(self, x):
        coded = self.coder(x)
        reconstructed = self.decoder(coded)
        return coded, reconstructed
