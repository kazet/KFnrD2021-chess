import re
import unittest

from data_compressor import extract_moves

path = ...  # path to pgn with games


class MyTestCase(unittest.TestCase):
    """Tests for function ``data_comressor.extract_moves``
    """

    def test_moves_number(self):
        """Test if all moves of a game was extracted
        """
        games = iter(extract_moves(path))
        with open(path) as file:
            for line in file:
                if line.startswith('1'):
                    match = re.search(r' \.\d+', line[::-1])  # Find last move number in format '<num>. '
                    moves_num = int(match.group()[:1:-1])*2 - 1
                    span = match.span()[0] * -1
                    if match.group()[:0:-1] + '.. ' in line or line[span + line[span:].find(' ') + 1].isalpha():
                        # Number is listed second time with '...' or after move there is another move, which starts
                        # with a letter (otherwise there is comment or result, which not starts with a letter)
                        moves_num += 1
                    try:
                        game = next(games)
                    except StopIteration:
                        return
                    self.assertEqual(moves_num, len(game), msg=line)


if __name__ == '__main__':
    unittest.main()
