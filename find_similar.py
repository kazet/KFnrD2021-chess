import torch
import sqlite3
from scipy.spatial.distance import cdist, cosine
import numpy as np
import json
import chess
import re
import time

import settings
from model import Autoencoder
from inference import Inference


def l1_loss(matrix, target):
    return [np.sum(np.abs(np.array(m) - np.array(target))) for m in matrix]


def nearest(matrix, target):
    return cdist(np.array(matrix), np.atleast_2d([target]))


def cosine_dist(matrix, target):
    return [cosine(m, target) for m in matrix]


similarity_functions = {
    "Closest vectors": nearest,
    "L1 Loos": l1_loss,
    "Cosine distance": cosine_dist,
}


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


def find_lowest(array, num, key=lambda i: i[1]):
    """
    Finds x smallest values ​​in a table.
    :param array: list of records
    :param num: number of records to be found
    :param key: sort function key
    :return: list of index and records
    """
    result = []
    for i, a in enumerate(array):
        if len(result) < num:
            result.append((i, a))
            result.sort(key=key)
        else:
            if result[-1][1] > a:
                result[-1] = (i, a)
                result.sort(key=key)
    return result


def find_game(idx, cursor):
    result = []
    for i in idx:
        data = cursor.execute(
            "SELECT positions_lite.Author, positions_lite.Number, positions_lite.Move FROM positions_lite WHERE (((positions_lite.ID)="
            + str(i[0])
            + "));"
        ).fetchall()[0]
        move = data[2]
        data = cursor.execute(
            "SELECT * FROM games_lite WHERE (((games_lite.Author)='"
            + data[0]
            + "') AND ((games_lite.Number)="
            + str(data[1])
            + "));"
        ).fetchall()
        result.append(list(data[0]) + [move])
    return result


def find_similar(fen, num=1, similarity_function=nearest):
    """
    Finds x smallest values ​​in a table.
    :param fen: String with chess position in Fen notation
    :param num: Number of games to be found
    :param similarity_function: Function that measures the similarity of vectors
    :return: list of games with similar positions
    """
    coder = Autoencoder(settings.BOARD_SHAPE, settings.LATENT_SIZE).to(
        settings.DEVICE
    )
    coder.load_state_dict(torch.load(settings.CODER_PATH, map_location=settings.DEVICE))
    coder = coder.coder
    coder.eval()
    inf = Inference(settings.DEVICE, coder)
    target = inf.predict([fen]).tolist()[0]
    conn = sqlite3.connect(settings.DATABASE)
    cursor = conn.cursor()
    cursor.execute("select Embeding FROM positions_lite")
    matrix = cursor.fetchall()
    matrix = [json.loads(x[0])[0] for x in matrix]
    scores = similarity_function(matrix, target)
    idx = find_lowest(scores, num)
    return Games(find_game(idx, cursor))


class Games:
    def __init__(self, data):
        """
        :param data: list from find_similar function
        """
        self.board = chess.Board()
        self.moves = []
        self.current_move = 0
        self.current_game = 0
        self.main_move = []
        self.info = []
        for game in data:
            info = {}
            for inf in game[2].split("|")[:-1]:
                try:
                    info[inf.split("'")[0][1:]] = inf.split("'")[1]
                except:
                    info[inf.split('"')[0][1:]] = inf.split('"')[1]
            self.info.append(info)
            self.moves.append(get_move(game[3]))
            self.main_move.append(game[4])
        self.set_board()

    def on_main_move(self):
        return self.current_move == self.main_move[self.current_game]

    def set_board(self):
        self.board.reset()
        for i in range(self.main_move[self.current_game]):
            self.board.push_san(self.moves[self.current_game][i])
        self.current_move = self.main_move[self.current_game]

    def next_game(self):
        self.current_game += 1
        if self.current_game >= len(self.moves):
            self.current_game = 0
        self.set_board()

    def last_game(self):
        self.current_game -= 1
        if self.current_game < 0:
            self.current_game = len(self.moves) - 1
        self.set_board()

    def next_move(self):
        if len(self.moves[self.current_game]) > self.current_move:
            self.board.push_san(self.moves[self.current_game][self.current_move])
            self.current_move += 1

    def last_move(self):
        if self.current_move > 0:
            self.board.pop()
            self.current_move -= 1

    def get_info(self):
        return self.info[self.current_game]

def pgn_games(pgn,n_game, player = "", start_pos = 0):
    result = []
    info = ""
    ID = 0
    for line in pgn :
        realLine = line.decode("utf-8", "backslashreplace")
        if(realLine[0] == "["):
            info += realLine[:-1]+"|"
        elif realLine[0] == "1":
            result.append([player,ID,info,realLine[:-1],start_pos])
            info = ""
            ID += 1
            if ID >= n_game:
                break
    return Games(result)



#with urlopen("https://lichess.org/api/games/user/sebb306?max=10") as pgn:
#    print(PgnGames(pgn,10))
