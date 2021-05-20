import torch
import pyodbc
from scipy.spatial.distance import cdist, cosine
import numpy as np
import json
import chess
import re

import settings
import model
from inference import Inference


def key(i):
    return i[1]

def l1_loss(matrix,target):
    return [np.sum(np.abs(np.array(m)-np.array(target))) for m in matrix]
    
def nearest(matrix,target):
    return cdist(matrix, np.atleast_2d([target]))
    
def cosine_dist(matrix,target):
    return [cosine(m, target) for m in matrix]
    
similarity_functions = {"Closest vectors":nearest,"L1 Loos":l1_loss,"Cosine distance":cosine_dist}

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


def find_lowest(array, num, key=key):
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
            "SELECT Embeding.Autor, Embeding.Number, Embeding.Move FROM Embeding WHERE (((Embeding.Identyfikator)="
            + str(i[0])
            + "));"
        ).fetchall()[0]
        move = data[2]
        data = cursor.execute(
            "SELECT * FROM Games_lite WHERE (((Games_lite.Autor)='"
            + data[0]
            + "') AND ((Games_lite.Number)="
            + str(data[1])
            + "));"
        ).fetchall()
        result.append(list(data[0]) + [move])
    return result


def find_similar(fen, num=1,similarity_function = nearest):
    """
    Finds x smallest values ​​in a table.
    :param fen: String with chess position in Fen notation
    :param num: Number of games to be found
    :param similarity_function: Function that measures the similarity of vectors
    :return: list of games with similar positions
    """
    coder = model.Coder(settings.BOARD_SHAPE, settings.LATENT_SIZE).to(settings.DEVICE)
    # coder.load_state_dict(torch.load(setting.CODER_PATH))
    coder.eval()
    inf = Inference(settings.DEVICE, coder)
    target = inf.predict([fen]).tolist()[0]
    conn = pyodbc.connect(settings.DATABASE)
    cursor = conn.cursor()
    cursor.execute("select Embeding FROM Embeding")
    matrix = cursor.fetchall()
    matrix = [json.loads(x[0])[0] for x in matrix]
    scores = similarity_function(matrix,target)
    idx = find_lowest(scores, num)
    return games(find_game(idx, cursor))


class games:
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
                info[inf.split("'")[0][1:]] = inf.split("'")[1]
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
