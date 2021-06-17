import chess
import chess.engine
import os
import math
import random
from time import time


def score_similarity(score1, score2, same_players=True, similarity_range=2000):
    """
    :param score1: score of first fen, in lan format, convert to str
    :param score1: score of second fen, in lan format, convert to str
    :param same_players: says whether in both cases,
    the move is for the player of the same color
    :param similarity_range: hyperparameter, int
    :return: similarity of score for given items in the range from -1 to 1
    """
    if score1[0] == "#" or score2[0] == "#":
        if score1[0] == "#" and score2[0] == "#":
            s1, s2 = int(score1[1:]), int(score2[1:])
            if same_players == False:
                s1 = -s1
            if (s1 < 0) != (s2 < 0):
                return -1
            return math.sqrt(min(s1, s2) / max(s1, s2))
        return -1
    s1, s2 = int(score1), int(score2)
    if same_players == False:
        s1 = -s1
    return 1 - min(2, max(0, abs(s1 - s2) / similarity_range))


def moves_similarity(pv1, pv2, board1, board2, depth=5):
    """
    :param pv1: list of moves from first fen, in class chess.Move
    :param pv1: list of moves from second fen, in class chess.Move
    :param board1: board with set first fen
    :param board1: board with set second fen
    :param depth: number of moves to analyze
    :return: similarity of moves for given items in the range from -1 to 1
    """
    result = 0
    length = min(len(pv1), len(pv2), depth)
    for move_idx in range(length):
        try:
            lan1, lan2 = board1.lan(pv1[move_idx]), board2.lan(pv2[move_idx])
            if (lan1[0].islower() and lan2[0].islower()) or lan1[0] == lan2[0]:
                # move with the same figure
                result += 0.5 * (length - move_idx)
            if ("x" in lan1 and "x" in lan2) or ("-" in lan1 and "-" in lan2):
                # both movements are or are not beatings
                result += 0.5 * (length - move_idx)

        except Exception:
            pass
    return result / (length / 2 * (length + 1))


def fen_to_coord(fen):
    fen_board = fen.split()[0]
    result = []
    pieces = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
    for piece in pieces:
        result.append([])
        for y, fen_row in enumerate(fen_board.split("/")):
            x = 0
            for fen_char in fen_row:
                if fen_char.isdigit():
                    x += int(fen_char)
                elif fen_char == piece:
                    x += 1
                    result[-1].append((x, y))
                else:
                    x += 1

    return result


def pieces_similarity(fen1, fen2):
    """
    :param fen1: first fen
    :param fen2: second fen
    :return: similarity of pieces for given fens in the range from -1 to 1
    """
    coordinates1 = fen_to_coord(fen1)
    coordinates2 = fen_to_coord(fen2)
    result = 0
    length = 0
    for n_piece in range(12):
        piece_result = 0
        for coord in coordinates1[n_piece]:
            if coord in coordinates2[n_piece]:
                piece_result += 1
        length += len(coordinates1[n_piece]) + len(coordinates2[n_piece]) - piece_result
        result += piece_result
    return result / length * 2 - 1


def predict_similarity(
    fen1,
    fen2,
    engine,
    moves_deep=5,
    time_limit=0.1,
    moves_similarity_weight=1,
    score_similarity_weight=1,
    pieces_similarity_weight=1,
    info1=None,
    info2=None,
):
    """
    :param fen1: first fen
    :param fen2: second fen
    :param engine: chess engine, in class chess.engine.SimpleEngine
    :param depth: number of moves to be analyzed in the move_similarity function
    :param time_limit: float of max time to analyze fen
    :param moves_similarity_weight,score_similarity_weight,pieces_similarity_weight:
    the weight of the result of a given similarity function
    :return: similarity given fens in the range from -1 to 1
    """
    board1 = chess.Board(fen1)
    board2 = chess.Board(fen2)
    if info1 == None:
        info1 = engine.analyse(
            board1, chess.engine.Limit(depth=moves_deep, time=time_limit)
        )
    if info2 == None:
        info2 = engine.analyse(
            board2, chess.engine.Limit(depth=moves_deep, time=time_limit)
        )
    try :
        ms = moves_similarity(info1["pv"], info2["pv"], board1, board2)
    except:
        ms = -1
    ss = score_similarity(
        str(info1["score"].relative),
        str(info2["score"].relative),
        ("w" in fen1) == ("w" in fen2),
    )
    ps = pieces_similarity(fen1, fen2)

    return (
        ms * moves_similarity_weight
        + ss * score_similarity_weight
        + ps * pieces_similarity_weight
    ) / (moves_similarity_weight + score_similarity_weight + pieces_similarity_weight)


if __name__ == "__main__":
    engine = chess.engine.SimpleEngine.popen_uci(
        os.path.join(os.getcwd(), "stockfish_13_win_x64", "stockfish_13_win_x64.exe")
    )
    with open("lichessTestFen.pgn", "r") as f:
        fens = f.read().split("\n")
        for i in range(252):
            print(
                predict_similarity(
                    fens[random.randint(0, 251)], fens[random.randint(0, 251)], engine
                )
            )
    engine.quit()
