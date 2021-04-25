import torch
from decouple import config

DEVICE = torch.device(config('DEVICE', default='cuda', cast=str))
BATCH_SIZE = config('BATCH_SIZE', default=64, cast=int)
LATENT_SIZE = config('LATENT_SIZE', default=16, cast=int)
LEARNING_RATE = config('LEARNING_RATE', default=2e-4)
SAVE_INTERVAL = config('SAVE_INTERVAL', default=10000, cast=int)
BOARD_SHAPE = config('BOARD_SHAPE', default=(17, 8, 8))
GENERATORS_NUMBER = config('GENERATORS_NUMBER', default=1, cast=int)
DATASETS = config('DATASETS')
