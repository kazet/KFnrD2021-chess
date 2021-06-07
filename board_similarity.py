import chess
import chess.engine
import os
import math
import random


def score_similarity(score1, score2, same_players=True, similarity_range=2000):
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
):
    board1 = chess.Board(fen1)
    board2 = chess.Board(fen2)
    info1 = engine.analyse(board1, chess.engine.Limit(time=time_limit))
    info2 = engine.analyse(board2, chess.engine.Limit(time=time_limit))
    ms = moves_similarity(info1["pv"], info2["pv"], board1, board2)
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
