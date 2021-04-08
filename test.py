from GeneratorPgn import Preprocessor, ArrayToFen, FENGenerator
import chess
import numpy as np
import torch.multiprocessing as mp
from collections import deque
import pickle

path_to_test_pgn = "lichessTestPgn.pgn"
path_to_test_fen = "lichessTestFen.pgn"
path_to_test_array = "lichessTestArray.dictionary"


class testIterator:
    def __init__(self):
        self.generator = FENGenerator(mp.Queue(maxsize=4), path_to_test_pgn)

    def __iter__(self):
        yield [self.generator.get_position()]


generator = FENGenerator(mp.Queue(maxsize=4), path_to_test_pgn)
answers = open(path_to_test_fen, "r")

index = 0
for answer in answers:
    index += 1
    data_from_generator = generator.get_position()
    assert data_from_generator == answer.replace(
        "\n", ""
    ), "mistake in FENGenerator class, in example " + str(index)


with open(path_to_test_array, "rb") as config_dictionary_file:
    answers = pickle.load(config_dictionary_file)

test_iterator = testIterator()
test_loader = Preprocessor(test_iterator)
index = 0
for answer in answers:
    index += 1
    for i, j in test_loader:
        assert (
            np.array(i["boards"]) == np.array(answer["boards"])
        ).all() == True, "mistake in Preprocessor class, in example " + str(index)
        assert ArrayToFen(i) == j, (
            "mistake in ArrayToFen function, in example "
            + str(index)
            + "\nIt should come out "
            + str(j)
            + " and it came out "
            + str(ArrayToFen(i))
        )
