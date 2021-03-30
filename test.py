import torch
import model
import settings
import loader

import numpy as np
import matplotlib.pyplot as plt


autoencoder = model.Autoencoder(settings.BOARD_SHAPE, settings.LATENT_SIZE).to(settings.DEVICE)
autoencoder.load_state_dict(torch.load('./records/model-25555-0.00482.pth'))

plt.ion()
plt.show()
fig, ((ax1, ax2, ax3),
      (ay1, ay2, ay3)) = plt.subplots(2, 3)

for X_batch_v in loader.StandardConvSuite(loader.Preprocessor(
        loader.Iterator(64, settings.STOCKFISH_PATH),
        settings.DEVICE)):
    _, X_rec_v = autoencoder(X_batch_v)
    for sample, sample2 in zip(X_batch_v, X_rec_v):
        ax2.set_title(f'min: {X_batch_v.min()}'
                      f'max: {X_batch_v.max()}'
                      f'mean: {X_batch_v.mean()}'
                      f'size: {X_batch_v.size()}')
        board_viz = (np.arange(1, 13)[:, None, None] * sample[:12].data.cpu().numpy()).mean(axis=0)
        en_passant_viz = sample[12].data.cpu().numpy()
        castlings_viz = sample[13:].data.cpu().numpy().mean(axis=0)
        ax1.imshow(board_viz, cmap='jet')
        ax2.imshow(en_passant_viz, vmin=0, vmax=1, cmap='jet')
        ax3.imshow(castlings_viz, vmin=0, vmax=1, cmap='jet')

        ay2.set_title(f'min: {X_rec_v.min()}'
                      f'max: {X_rec_v.max()}'
                      f'mean: {X_rec_v.mean()}'
                      f'size: {X_rec_v.size()}')
        board_viz = (np.arange(1, 13)[:, None, None] * sample2[:12].data.cpu().numpy()).mean(axis=0)
        en_passant_viz = sample2[12].data.cpu().numpy()
        castlings_viz = sample2[13:].data.cpu().numpy().mean(axis=0)
        ay1.imshow(board_viz, cmap='jet')
        ay2.imshow(en_passant_viz, vmin=0, vmax=1, cmap='jet')
        ay3.imshow(castlings_viz, vmin=0, vmax=1, cmap='jet')

        plt.draw()
        plt.pause(.01)
