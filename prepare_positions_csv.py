import torch
import pyodbc
import chess
import csv
import re
import os

import settings
import model
from inference import Inference

"""
preparing a csv file from embeding the chess positions from 
the games of the masters from the website "https://www.pgnmentor.com/players/"
"""


def get_move(line):
    line = str(line)
    game = []  # list of moves in pgn notation
    for move in re.findall(
        r"[^{\[.}\]]+ ", line.replace("?", "").replace("!", "")
    ):  # Extract all moves without comments
        game.extend(
            move.strip().split(" ")
        )  # Sometimes one string contains two moves with space between
    return game


coder = model.Autoencoder(settings.BOARD_SHAPE, settings.LATENT_SIZE).to(settings.DEVICE)
coder.load_state_dict(torch.load(settings.CODER_PATH, map_location=settings.DEVICE))
coder = coder.coder
coder.eval()
inf = Inference(settings.DEVICE, coder)
csv_name = "positions_lite.csv"
ID = 0

with open(csv_name, "w", newline="") as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerow(["ID","Author", "Number", "Move", "Embeding"])
    games_csv = open(os.path.join(os.getcwd(),'games_lite.csv'))
    for row in games_csv:
        try :
            data = row[:-2].split(";")
            board = chess.Board()
            for idx, move in enumerate(get_move(data[3])):
                board.push_san(move)
                tensor = inf.predict([board.fen()])
                writer.writerow([ID, data[0], int(data[1]), idx, str(tensor.tolist())])
                ID += 1
        except:
            print("Error on "+data[0],data[1])
