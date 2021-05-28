import bz2
import os
import re
from typing import List, Iterator, Optional


def extract_games(path: str) -> Iterator[List[str]]:
    """Generator, whith opens file on given path and proccesses every game from it
    to list of moves in pgn notation

    :param path: path to the file with games

    :yield: Game as moves strings (listed)
    """
    if path.endswith('.bz2'):
        opn = bz2.open  # open file with decompressing if needed
    else:
        opn = open
    with opn(path, 'r') as file:
        for line in file:
            if isinstance(line, bytes):
                line = line.decode('utf-8')  # line can be bytes, so it should be converted
            if line.startswith('1'):  # Moves are in one line starting with 1
                game = []  # list of moves in pgn notation
                for move in re.findall(
                        r' [^{\[.}\]]+ ', line.replace('?', '').replace('!', '')
                ):  # Extract all moves without comments
                    game.extend(move.strip().split(' '))  # Sometimes one string contains two moves with space between
                yield game


def preprocess_data(path: str, output: Optional[str] = None, *, delete_original: bool = True) -> str:
    """Extracts moves from file on given path using :func extract_games: and saves it to compressed file
     in .bz2 format to original location or ``output`` if provided

    :param path: path to a file
    :param output: optional path to save location
    :param delete_original: wheather to delete original uncompressed file, default `True`

    :return: path to compressed file
    """
    compressor = bz2.BZ2Compressor()
    output = output if output else os.path.join(os.path.dirname(path), 'proc_' + os.path.basename(path))
    # modifying path is necessary, because otherwise we read and write to the same file
    if not output.endswith('.bz2'):
        output += '.bz2'
    with open(output, 'wb') as f:
        for game in extract_games(path):  # extract and compress games one by one
            game_str = bytes(' '.join(game) + '\n', encoding='utf-8')
            data = compressor.compress(game_str)
            f.write(data)
        f.write(compressor.flush())

    if delete_original:
        os.remove(path)
    return output
