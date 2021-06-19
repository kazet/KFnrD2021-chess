import torch
from torch import optim
from tensorboardX import SummaryWriter
import torch.nn.functional as F

import numpy as np
import settings

import vae_model
import loader

if __name__ == '__main__':
    torch.random.manual_seed(42)
    np.random.seed(42)
    # Defining network and optimizers
    autoencoder = vae_model.VAE(settings.BOARD_SHAPE, settings.LATENT_SIZE).to(settings.DEVICE)
    optimizer = optim.Adam(autoencoder.parameters(), settings.LEARNING_RATE)
    data_paths = loader.getListOfDataset(settings.DATASETS, settings.GENERATORS_NUMBER)
    iterator = loader.PGNIterator(settings.BATCH_SIZE, data_paths)
    preprocessor = loader.Preprocessor(iterator, settings.DEVICE)
    data_source = loader.StandardConvSuite(preprocessor)
    writer = SummaryWriter(comment='VariationalAutoencoder')
    best_loss = np.inf

    for idx, X_batch_v in enumerate(data_source):
        # Training models
        optimizer.zero_grad()
        x_hat, mean, log_var = autoencoder(X_batch_v)
        loss_reconstruction_v, losses = vae_model.loss_reconstruction(X_batch_v, x_hat, mean, log_var)
        loss_reconstruction_v.backward()
        optimizer.step()

        # Logging to tensorboard
        writer.add_scalar('Loss', loss_reconstruction_v.item(), idx,
                          summary_description='Loss used for training, sum of reproduction loss and KL divergence')
        writer.add_scalar('Losses/reproduction_loss', losses[0].item(), idx)
        writer.add_scalar('Losses/KL divergence', losses[1].item(), idx)
        writer.add_scalar('Losses/L1 loss', F.l1_loss(X_batch_v, x_hat).item(), idx)
        writer.add_scalar('Losses/Smooth l1 loss', F.smooth_l1_loss(X_batch_v, x_hat).item(), idx)
        writer.add_scalar('Losses/MSE loss', F.mse_loss(X_batch_v, x_hat).item(), idx)

        # Saving models and logging to console
        print(idx, loss_reconstruction_v.item())
        if not idx % settings.SAVE_INTERVAL:
            torch.save(autoencoder.state_dict(),
                       f'vae/checkpoints/model-{idx}-{np.round(loss_reconstruction_v.item(), 6)}.pth')
            print('SAVED MODEL TO CHECKPOINTS')
        if loss_reconstruction_v.item() < best_loss:
            best_loss = loss_reconstruction_v.item()
            torch.save(autoencoder.state_dict(),
                       f'vae/records/model-{idx}-{np.round(loss_reconstruction_v.item(), 6)}.pth')
            print('SAVED MODEL TO RECORDS')
