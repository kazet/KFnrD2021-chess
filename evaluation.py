import torch
import numpy as np
from scipy.spatial.distance import cdist
import chess.engine
from os import path, getcwd
from random import randint

import settings
from model import Autoencoder
from inference import Inference
from board_similarity import predict_similarity


def nearest(matrix, target):
    return cdist(np.array(matrix), np.atleast_2d([target]))


def model_similarest_position(coder_inference, similarity_function, fens):
    scores = []
    matrix = coder_inference.predict(fens).tolist()
    for j in range(len(matrix)):
        scores.append(
            np.argmin(similarity_function(matrix[:j] + matrix[j + 1 :], matrix[j]))
        )
    return scores


def manual_similarest_position(
    engine, similarity_function, fens, moves_deep=5, time_limit=0.1
):
    scores = []
    infos = [
        engine.analyse(
            chess.Board(fen), chess.engine.Limit(depth=moves_deep, time=time_limit)
        )
        for fen in fens
    ]
    for i in range(len(fens)):
        matrix = []
        for j in range(len(fens)):
            if j != i:
                matrix.append(
                    similarity_function(
                        fens[i], fens[j], engine, info1=infos[i], info2=infos[j]
                    )
                )
        scores.append(matrix.index(max(matrix)))
    return scores


def model_score(
    coder_inference,
    engine,
    model_similarity_function,
    manual_similarity_function,
    fens,
    batch_size=20,
    print_progress=False,
):
    """
    He takes n fen and checks for each one the most similar fen of the rest. 
    the results are compared with thats of the model
    """
    c_fens = fens.copy()
    n_correct_predict = 0
    n_predict = int(len(c_fens) / batch_size) * batch_size
    if print_progress:
        print("Progress : 0%", end="")
    while len(c_fens) >= batch_size:
        fenset = [c_fens.pop(randint(0, len(c_fens) - 1)) for j in range(batch_size)]
        model_predict = model_similarest_position(
            coder_inference, model_similarity_function, fenset
        )
        manual_predict = manual_similarest_position(
            engine, manual_similarity_function, fenset
        )
        for i in range(batch_size):
            if model_predict[i] == manual_predict[i]:
                n_correct_predict += 1
        if print_progress:
            progress = int((1 - len(c_fens) / n_predict) * 100)
            print("\r", "Progress : " + str(progress) + "%", end="")
    if print_progress:
        print("\r", "Progres : 100%, End")
    return n_correct_predict / n_predict


if __name__ == "__main__":
    coder = Autoencoder(settings.BOARD_SHAPE, settings.LATENT_SIZE).to(settings.DEVICE)
    coder.load_state_dict(torch.load(settings.CODER_PATH, map_location=settings.DEVICE))
    coder = coder.coder
    coder.eval()
    inf = Inference(settings.DEVICE, coder)

    engine = chess.engine.SimpleEngine.popen_uci(
        path.join(getcwd(), "stockfish_13_win_x64", "stockfish_13_win_x64.exe")
    )

    fens = None
    with open("lichessTestFen.pgn", "r") as f:
        fens = f.read().split("\n")

    tests_fens = [
        "rnbqkbnr/ppp1pppp/8/8/2pPP3/8/PP3PPP/RNBQKBNR b KQkq - 0 3",
        "rnbqkbnr/ppp1pp1p/6p1/8/2pPP3/8/PP3PPP/RNBQKBNR w KQkq - 0 4",
        "rnbqkbnr/ppp1pp1p/6p1/8/2BPP3/8/PP3PPP/RNBQK1NR b KQkq - 0 4",
        "rnbqk1nr/ppp1ppbp/6p1/8/2BPP3/8/PP3PPP/RNBQK1NR w KQkq - 1 5",
        "rnbqk1nr/ppp1ppbp/6p1/8/2BPP3/8/PP2NPPP/RNBQK2R b KQkq - 2 5",
        "rnbqk2r/ppp1ppbp/5np1/8/2BPP3/8/PP2NPPP/RNBQK2R w KQkq - 3 6",
        "rnbqk2r/ppp1ppbp/5np1/8/2BPP3/2N5/PP2NPPP/R1BQK2R b KQkq - 4 6",
        "rnbq1rk1/ppp1ppbp/5np1/8/2BPP3/2N5/PP2NPPP/R1BQK2R w KQ - 5 7",
        "rnbq1rk1/ppp1ppbp/5np1/8/2BPP3/2N5/PP2NPPP/R1BQ1RK1 b - - 6 7",
        "rnbq1rk1/ppp2pbp/4pnp1/8/2BPP3/2N5/PP2NPPP/R1BQ1RK1 w - - 0 8",
        "rnbq1rk1/ppp2pbp/4pnp1/8/2BPP3/2N1B3/PP2NPPP/R2Q1RK1 b - - 1 8",
    ]

    print(model_score(inf, engine, nearest, predict_similarity, fens, 5))

    engine.quit()
