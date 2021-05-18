import torch
import pyodbc
from scipy.spatial.distance import cdist
import numpy as np
import json

import settings
import model
from inference import Inference


def key(i):
    return i[1]


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


def find_similar(fen, num=1):
    """
    Finds x smallest values ​​in a table.
    :param fen: String with chess position in Fen notation
    :param num: Number of games to be found
    :return: list of games with similar positions
    """
    coder = model.Coder(settings.BOARD_SHAPE, settings.LATENT_SIZE).to(settings.DEVICE)
    coder.eval()
    inf = Inference(settings.DEVICE, coder)
    target = inf.predict([fen]).tolist()[0]
    conn = pyodbc.connect(
        r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ="
        + settings.ACCES_DATABASE
        + ";"
    )
    cursor = conn.cursor()
    cursor.execute("select Embeding FROM Embeding")
    matrix = cursor.fetchall()
    matrix = [json.loads(x[0])[0] for x in matrix]
    scores = cdist(matrix, np.atleast_2d([target]))
    idx = find_lowest(scores, num)
    return find_game(idx, cursor)


if __name__ == "__main__":
    print(find_similar("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2))
