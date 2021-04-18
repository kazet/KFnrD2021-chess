import bz2
import os
import re
import shutil
import unittest
from typing import Iterator, List, IO

from preprocessor import extract_games, preprocess_data

# example pgns with games, shortened to fit into repo
path = os.path.join(os.path.dirname(__file__), 'example_games.pgn')
compressed_path = os.path.join(os.path.dirname(__file__), 'example_games.pgn.bz2')


class GamesExtractionTest(unittest.TestCase):
    """Tests for function :func extract_games:
    """

    def _moves_number_helper(self, file: IO, games: Iterator[List[str]]) -> None:
        """Helper function for tests (DRY purpose)

        :param file: open file with game data
        :param games: iterator returned by :func extract_games:

        :return: None
        """
        for line in file:
            line = str(line)  # line can be bytes, so it should be converted
            if line.startswith('1'):
                match = re.search(r' \.\d+', line[::-1])  # Find last move number in format '<num>. '
                moves_num = int(match.group()[:1:-1]) * 2 - 1
                span = match.span()[0] * -1
                if match.group()[:0:-1] + '.. ' in line or line[span + line[span:].find(' ') + 1].isalpha():
                    # Number is listed second time with '...' or after move there is another move, which starts
                    # with a letter (otherwise there is comment or result, which not starts with a letter)
                    moves_num += 1
                try:
                    game = next(games)
                except StopIteration:
                    game = ''  # test if all games were extracted
                self.assertEqual(moves_num, len(game), msg=line)

    def test_moves_number(self):
        """Test if all moves of each game was extracted
        """
        games = extract_games(path)
        with open(path) as file:
            self._moves_number_helper(file, games)

    def test_moves_number_compressed(self):
        """Test if all moves of each game was extracted, when passed compressed file
        """
        games = extract_games(compressed_path)
        with bz2.open(compressed_path) as file:
            self._moves_number_helper(file, games)


class CompressingTest(unittest.TestCase):
    """Test for :func preprocess_data:
    """

    def _test_compressing_helper(self, file_path: str) -> None:
        """Helper funcion for tests (DRY purpose)

        :param file_path: path to preprocesssed file
        """
        games = extract_games(path)
        text = '\n'.join([' '.join(game) for game in games]) + '\n'
        text = bytes(text, encoding='utf-8')
        with open(file_path, 'rb') as f:
            decomp = bz2.decompress(f.read())
        os.remove(file_path)
        self.assertEqual(text, decomp)  # compare file originally constructed with decompressed

    def test_compressing_with_output(self):
        """Test wheather compressed data would be the same after decompression if `output` parameter is passed
        """
        file_path = preprocess_data(path, '_temp', delete_original=False)
        self._test_compressing_helper(file_path)

    def test_compressing_without_output(self):
        """Test wheather compressed data would be the same after decompression if `output` parameter is not
        passed"""
        new_path = os.path.join(os.path.dirname(path), '_temp_' + os.path.basename(path))
        shutil.copyfile(path, new_path)  # copy file in order not to modify original file
        file_path = preprocess_data(new_path)
        self._test_compressing_helper(file_path)


if __name__ == '__main__':
    unittest.main()
