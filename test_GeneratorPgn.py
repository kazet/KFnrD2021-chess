from GeneratorPgn import Preprocessor, ArrayToFen, FENGenerator, PGNIterator
import chess
import numpy as np
import torch.multiprocessing as mp
from collections import deque
import pickle
import unittest


path_to_test_pgn = "lichessTestPgn.pgn"
path_to_test_fen = "lichessTestFen.pgn"
path_to_test_array = "lichessTestArray.dictionary"


class testIterator:
    def __init__(self):
        self.generator = FENGenerator(mp.Queue(maxsize=4), path_to_test_pgn)

    def __iter__(self):
        yield [self.generator.get_position()]

    def close(self):
        self.generator.closeFile()


class TestSum(unittest.TestCase):
    def test_FENGenerator(self):
        self.generator = FENGenerator(mp.Queue(maxsize=4), path_to_test_pgn)
        self.answers = open(path_to_test_fen, "r")
        self.index = 0
        for answer in self.answers:
            self.index += 1
            self.data_from_generator = self.generator.get_position()
            self.assertEqual(
                self.data_from_generator,
                answer.replace("\n", ""),
                "mistake in FENGenerator class, in example " + str(self.index),
            )
        self.answers.close()
        self.generator.closeFile()

    def test_Preprocessor(self):
        with open(path_to_test_array, "rb") as config_dictionary_file:
            self.answers = pickle.load(config_dictionary_file)
        self.test_iterator = testIterator()
        self.test_loader = Preprocessor(self.test_iterator)
        self.index = 0
        for answer in self.answers:
            self.index += 1
            for data, fen in self.test_loader:
                for key in data:
                    self.assertEqual(
                        (np.array(data[key]) == np.array(answer[key])).all(),
                        True,
                        "mistake in Preprocessor class, in example "
                        + str(self.index)
                        + " in key "
                        + key,
                    )
                self.assertEqual(
                    ArrayToFen(data),
                    fen,
                    (
                        "mistake in ArrayToFen function, in example "
                        + str(self.index)
                        + "\nIt should come out "
                        + str(data)
                        + " and it came out "
                        + str(ArrayToFen(data))
                    ),
                )
        self.test_loader.iterator.close()

    def test_ArrayToFen(self):
        self.test_loader = Preprocessor(PGNIterator(1, [path_to_test_pgn], 1, 1))
        self.index = 0
        for data, fen in self.test_loader:
            self.index += 1
            self.assertEqual(
                ArrayToFen(data),
                fen,
                (
                    "mistake in ArrayToFen function.\nIt should come out "
                    + str(data)
                    + " and it came out "
                    + str(ArrayToFen(data))
                ),
            )
            if self.index == 200:
                self.test_loader.close()
                break


if __name__ == "__main__":
    unittest.main()
