import torch
import torch.nn as nn
import torch.nn.functional as F

from settings import DEVICE


class Coder(nn.Module):
    def __init__(self, in_shape, latent_size):
        super(Coder, self).__init__()
        self.conv_1 = nn.Conv2d(in_shape[0], 32, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0))
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), padding=(1, 1))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 256)
        self.fc_mean = nn.Linear(256, latent_size)
        self.fc_log_var = nn.Linear(256, latent_size)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.conv_1(x))
        x = self.leaky_relu(self.conv_2(x))
        x = self.leaky_relu(self.conv_3(x))
        batch = x.shape[0]
        x = self.pool(x).reshape(batch, -1)

        hidden = self.fc(x)
        mean = self.fc_mean(hidden)
        log_var = self.fc_log_var(hidden)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, out_shape, latent_size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_size, 128)
        self.conv_1 = nn.ConvTranspose2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_2 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0))
        self.conv_3 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv_4 = nn.ConvTranspose2d(32, out_shape[0], kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 1, 1)
        x = self.leaky_relu(self.conv_1(x))
        x = self.leaky_relu(self.conv_2(x))
        x = self.leaky_relu(self.conv_3(x))
        x = self.sigmoid(self.conv_4(x))
        return x


class VAE(nn.Module):
    def __init__(self, input_shape, latent_size):
        super(VAE, self).__init__()
        self.coder = Coder(input_shape, latent_size)
        self.decoder = Decoder(input_shape, latent_size)

    def reparametrize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std).to(DEVICE)
        z = mean + std*eps
        return z

    def forward(self, x):
        mean, log_var = self.coder(x)
        z = self.reparametrize(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var


def loss_reconstruction(x, x_hat, mean, log_var):
    reproduction_loss = F.binary_cross_entropy(x_hat, x)
    KLD = - 0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD, (reproduction_loss, KLD)


if __name__ == '__main__':  # testing output shape
    t = torch.rand(1, 17, 8, 8)
    model = VAE((17, 8, 8), 16)
    model.eval()
    t1 = model(t)
    print(t1[0].shape)
