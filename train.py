import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import numpy as np
import settings
import model
import loader


if __name__ == '__main__':
    torch.random.manual_seed(42)
    np.random.seed(42)
    # Defining network and optimizers
    autoencoder = model.Autoencoder(settings.BOARD_SHAPE, settings.LATENT_SIZE).to(settings.DEVICE)
    optimizer = optim.Adam(autoencoder.parameters(), settings.LEARNING_RATE)
    data_paths = loader.getListOfDataset(settings.DATASETS,settings.GENERATORS_NUMBER)
    iterator = loader.PGNIterator(settings.BATCH_SIZE, data_paths)
    preprocessor = loader.Preprocessor(iterator, settings.DEVICE)
    data_source = loader.StandardConvSuite(preprocessor)
    writer = SummaryWriter(comment='ResidualAutoencoder-kernels:(SLOW-PROGRESSIVE)-layers(64|128|256|16)-dropout(.4|.2)')
    best_loss = np.inf

    for idx, X_batch_v in enumerate(data_source):
        # Progressive growth
        autoencoder.alpha_ascent(1./1000000.)

        # Training models
        optimizer.zero_grad()
        _, reconstruction = autoencoder(X_batch_v)
        loss_reconstruction_v = F.mse_loss(X_batch_v, reconstruction)
        loss_reconstruction_v.backward()
        optimizer.step()

        # Logging to tensorboard
        writer.add_scalar('loss_reconstruction', loss_reconstruction_v.item(), idx)
        writer.add_scalar('alpha', autoencoder.coder.conv_in.alpha, idx)
        writer.add_scalar('num_layers', autoencoder.coder.conv_in.active_layers, idx)

        # Saving models and logging to console
        print(loss_reconstruction_v)
        if not idx % settings.SAVE_INTERVAL:
            torch.save(autoencoder.state_dict(), f'checkpoints/model-{idx}-{np.round(loss_reconstruction_v.item(), 6)}.pth')
            print('SAVED MODEL TO CHECKPOINTS')
        if loss_reconstruction_v.item() < best_loss:
            best_loss = loss_reconstruction_v.item()
            torch.save(autoencoder.state_dict(), f'records/model-{idx}-{np.round(loss_reconstruction_v.item(), 6)}.pth')
            print('SAVED MODEL TO RECORDS')
