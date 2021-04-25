import chess
import numpy as np
import torch.multiprocessing as mp
from collections import deque
import pickle
import unittest
import torch

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from loader import Preprocessor, FENGenerator, PGNIterator, StandardConvSuite 

path_to_test_pgn = f"lichessTestPgn.pgn"
path_to_test_fen = f"lichessTestFen.pgn"
path_to_test_array = f"lichessTestArray.dictionary"
path_to_test_tensor = f"lichessTestTensor.dictionary"


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
        self.test_loader = Preprocessor(self.test_iterator,torch.device("cpu"))
        self.index = 0
        for answer in self.answers:
            self.index += 1
            for data in self.test_loader:
                for key in data:
                    self.assertEqual(
                        (np.array(data[key]) == np.array(answer[key])).all(),
                        True,
                        "mistake in Preprocessor class, in example "
                        + str(self.index)
                        + " in key "
                        + key,
                    )
                
        self.test_loader.iterator.close()

    def test_StandardConvSuite(self):
        with open(path_to_test_tensor, "rb") as config_dictionary_file:
            self.answers = pickle.load(config_dictionary_file)
        self.test_iterator = testIterator()
        self.test_loader = StandardConvSuite(Preprocessor(self.test_iterator,torch.device("cpu")))
        self.index = 0
        for answer in self.answers:
            self.index += 1
            for data in self.test_loader:                
                self.assertEqual(
                    (np.array(data) == np.array(answer)).all(),
                    True,
                    "mistake in StandardConvSuite class, in example "
                    + str(self.index)
                )
                
        self.test_loader.preprocessor.iterator.close()
        


if __name__ == "__main__":
    unittest.main()
