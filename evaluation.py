import torch
import numpy as np
from scipy.spatial.distance import cdist
import chess.engine
from os import path, getcwd, listdir
from random import randint
from copy import deepcopy

import settings
from model import Autoencoder
from inference import Inference
from board_similarity import predict_similarity


def nearest(matrix, target):
    return cdist(matrix, np.atleast_2d([target]))


def model_similarest_position(coder_inference, similarity_function, fens):
    scores = []
    matrix = np.array(coder_inference.predict(fens).tolist())
    for j in range(len(matrix)):
        scores.append(
            np.argmin(similarity_function(np.concatenate((matrix[:j] , matrix[j + 1 :]), axis=0), matrix[j]))
        )
    return scores


def manual_similarest_position(
    engine, similarity_function, fens, weight, moves_deep=5, time_limit=0.1,
):
    scores = []
    infos = [
        engine.analyse(
            chess.Board(fen), chess.engine.Limit(depth=moves_deep, time=time_limit)
        )
        for fen in fens
    ]
    for i in range(len(fens)):
        matrix = [[] for h in range(4)]
        for j in range(len(fens)):
            if j != i:
                results = similarity_function(
                    fens[i], fens[j], engine, info1=infos[i], info2=infos[j], weight = weight,give_all = True,
                )
                for g, result in enumerate(results):
                    matrix[g].append(result)
                
        scores.append([m.index(max(m)) for m in matrix])
    return scores


def model_score(
    coder_inference,
    engine,
    model_similarity_function,
    manual_similarity_function,
    fens,
    batch_size=20,
    print_progress=False,
    manual_weight = [1,1,1],
):
    """
    He takes n fen and checks for each one the most similar fen of the rest. 
    the results are compared with thats of the model
    """
    c_fens = fens.copy()
    n_correct_predicts = [0,0,0,0]
    n_predict = int(len(c_fens) / batch_size) * batch_size
    if print_progress:
        print("Progress : 0%", end="")
    while len(c_fens) >= batch_size:
        fenset = [c_fens.pop(randint(0, len(c_fens) - 1)) for j in range(batch_size)]
        model_predict = model_similarest_position(
            coder_inference, model_similarity_function, fenset
        )
        manual_predict = manual_similarest_position(
            engine, manual_similarity_function, fenset, manual_weight,
        )
        for i in range(batch_size):
            for j,predict in enumerate(manual_predict[i]):
                if predict == model_predict[i]:
                    n_correct_predicts[j] += 1
        if print_progress:
            progress = int((1 - len(c_fens) / n_predict) * 100)
            print("\r", "Progress : " + str(progress) + "%", end="")
    if print_progress:
        print("\r", "Progres : 100%, End")
    return [n_correct_predict / n_predict for n_correct_predict in n_correct_predicts]

def opening_score(coder_inference,similarity_function,files_dir,batch_size, max_test = None):
    openings = listdir(files_dir)
    all_tests = 0
    correct_test = 0
    while len(openings) >= batch_size:
        openings_dirs = [openings.pop(randint(0,len(openings)-1)) for i in range(batch_size)]
        openings_fens = []
        for dir in openings_dirs:
            with open(path.join(files_dir,dir),"r") as f:
                openings_fens.append(f.read().split("\n")[:-1])
        openings_matrix = [coder_inference.predict(fens).tolist() for fens in openings_fens]
        for main_idx in range(len(openings_matrix)):
            main_matrix = deepcopy(openings_matrix[main_idx])
            while len(main_matrix) >= 2:
                target = main_matrix.pop(randint(0,len(main_matrix)-1))
                matrix = [ main_matrix.pop(randint(0,len(main_matrix)-1)) ]
                for idx in range(len(openings_matrix)):
                    if idx == main_idx:
                        continue
                    matrix.append(openings_matrix[idx][randint(0,len(openings_matrix[idx])-1)])
                result = np.argmin(similarity_function(np.array(matrix),target))
                all_tests += 1
                if result == 0:
                    correct_test += 1
        if max_test!= None and max_test<= all_tests:
            break
    return correct_test / all_tests

if __name__ == "__main__":
    coder = Autoencoder(settings.BOARD_SHAPE, settings.LATENT_SIZE).to(settings.DEVICE)
    coder.load_state_dict(torch.load(settings.CODER_PATH, map_location=settings.DEVICE))
    coder = coder.coder
    coder.eval()
    inf = Inference(settings.DEVICE, coder)
    
    class RandomEmbeding():
        def predict(self,y):
            return torch.rand(len(y),16)
    inf3 = RandomEmbeding()
    
    coder2 = Autoencoder(settings.BOARD_SHAPE, settings.LATENT_SIZE).to(settings.DEVICE)
    coder2 = coder2.coder
    coder2.eval()
    inf2 = Inference(settings.DEVICE, coder2)

    engine = chess.engine.SimpleEngine.popen_uci(
        path.join(getcwd(), "stockfish_13_win_x64", "stockfish_13_win_x64.exe")
    )

    print(opening_score(inf,nearest,path.join(getcwd(),"openingsFen"),11,1000))
    print(opening_score(inf2,nearest,path.join(getcwd(),"openingsFen"),11,1000))
    print(opening_score(inf3,nearest,path.join(getcwd(),"openingsFen"),11,1000))

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

    print(model_score(inf, engine, nearest, predict_similarity, fens, 11))
    print(model_score(inf2, engine, nearest, predict_similarity, fens, 11))

    engine.quit()
