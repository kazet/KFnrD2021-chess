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
    writer = SummaryWriter(comment='ResidualAutoencoder')
    best_loss = np.inf

    for idx, X_batch_v in enumerate(data_source):
        # Training models
        optimizer.zero_grad()
        _, reconstruction = autoencoder(X_batch_v)
        loss_reconstruction_v = F.mse_loss(X_batch_v, reconstruction)
        loss_reconstruction_v.backward()
        optimizer.step()

        # Logging to tensorboard
        writer.add_scalar('loss_reconstruction', loss_reconstruction_v.item(), idx)
        writer.add_scalars('noises',
                           {
                               'coder-layer-1-w': autoencoder.coder.nn[-6].weight_noise.data.mean(),
                               'coder-layer-2-w': autoencoder.coder.nn[-4].weight_noise.data.mean(),
                               'decoder-layer-1-w': autoencoder.decoder.nn[0].weight_noise.data.mean(),
                               'decoder-layer-2-w': autoencoder.decoder.nn[2].weight_noise.data.mean(),
                               'coder-layer-1-b': autoencoder.coder.nn[-6].bias_noise.data.mean(),
                               'coder-layer-2-b': autoencoder.coder.nn[-4].bias_noise.data.mean(),
                               'decoder-layer-1-b': autoencoder.decoder.nn[0].bias_noise.data.mean(),
                               'decoder-layer-2-b': autoencoder.decoder.nn[2].bias_noise.data.mean()
                           }, idx)
        writer.add_scalar('noise_gate', autoencoder.coder.nn[-3].noise.data.mean(), idx)

        # Saving models and logging to console
        print(idx, loss_reconstruction_v)
        if not idx % settings.SAVE_INTERVAL:
            torch.save(autoencoder.state_dict(),
                       f'autoencoder/checkpoints/model-{idx}-{np.round(loss_reconstruction_v.item(), 6)}.pth')
            print('SAVED MODEL TO CHECKPOINTS')
        if loss_reconstruction_v.item() < best_loss:
            best_loss = loss_reconstruction_v.item()
            torch.save(autoencoder.state_dict(),
                       f'autoencoder/records/model-{idx}-{np.round(loss_reconstruction_v.item(), 6)}.pth')
            print('SAVED MODEL TO RECORDS')
