import bz2
import re
from typing import List, Optional


def extract_moves(path: str) -> List[List[str]]:
    """Opens file on given path and proccesses every game from it to list of moves in pgn notation

    :param path: path to the file with games
    :return: List of games consisting of moves strings
    """
    games = []  # list of all games in file
    with open(path) as file:
        for line in file:
            if line.startswith('1'):  # Moves are in one line starting with 1
                game = []  # list of moves in pgn notation
                for move in re.findall(
                        r' [^{\[.}\]]+ ', line.replace('?', '').replace('!', '')
                ):  # Extract all moves without comments
                    game.extend(move.strip().split(' '))  # Sometimes one string contains two moves with space between
                games.append(game)
        return games


def compress_file(path: str, output: Optional[str] = None) -> None:
    """Compresses file on given path and saves in .bz2 format in original location or ``output`` if provided

    :param path: path to a file
    :param output: optional path to save location
    :return: None
    """
    comp = extract_moves(path)
    comp = [' '.join(p) for p in comp]
    comp_str = '\n'.join(comp)
    output = (output if output else path) + '.bz2'

    with open(output, 'wb') as file:
        file.write(bz2.compress(comp_str.encode('utf-8')))
