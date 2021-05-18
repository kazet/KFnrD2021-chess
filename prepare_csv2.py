import torch
import pyodbc
import chess
import csv
import re

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


coder = model.Coder(settings.BOARD_SHAPE, settings.LATENT_SIZE).to(settings.DEVICE)
# coder.load_state_dict(torch.load(setting.CODER_PATH))
coder.eval()
inf = Inference(settings.DEVICE, coder)
csv_name = "embeding_lite2.csv"

with open(csv_name, "w", newline="") as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerow(["Autor", "Number", "Move", "Embeding"])
    conn = pyodbc.connect(
        r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ="
        + settings.ACCES_DATABASE
        + ";"
    )
    cursor = conn.cursor()
    cursor.execute("select * FROM Games_lite")
    for row in cursor.fetchall():
        board = chess.Board()
        for idx, move in enumerate(get_move(row[3])):
            board.push_san(move)
            tensor = inf.predict([board.fen()])
            writer.writerow([row[0], row[1], idx, str(tensor.tolist())])
