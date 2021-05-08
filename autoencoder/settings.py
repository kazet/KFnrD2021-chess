import torch

STOCKFISH_PATH = 'lib/stockfish_13_linux_x64_bmi2'
DEVICE = torch.device('cuda')
BATCH_SIZE = 64
LATENT_SIZE = 16
LEARNING_RATE = 2e-4
SAVE_INTERVAL = 10000
BOARD_SHAPE = (17, 8, 8)
