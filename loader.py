import chess
import bz2
import torch
import torch.multiprocessing as mp
from collections import deque
import numpy as np
import random
import os
from urllib.request import urlopen


# initial setup that looks ugly
FIRST_CORDS = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
SECOND_CORDS = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7}
PIECES = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
          'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}


class FENGenerator:
    def __init__(self, replay_queue: mp.Queue, pgn_path):

        """
        An iterable class that takes game saves and converts them into a chessboard position record in fen notation.
        :param replay_queue: target queue for samples
        :param pgn_paths: list of paths to data files
        """

        self.replay_queue = replay_queue
        self.error_tolerance=2
        self.current_path_index = (
            0  # index of the path to the currently used data file in the pgn_paths list
        )
        self.current_line = (
            []
        )  # currently used line from the data file, representing one game
        self.board = chess.Board()  # the chessboard on which the moves are made
        if isinstance(pgn_path,str):
            self.pgn_paths = [pgn_path]
        else:
            self.pgn_paths = pgn_path
        # pgn_file - currently used data file
        self.pgn_file = None
        self.get_new_file(0)

    def closeFile(self):
        if self.pgn_file != None:
            self.pgn_file.close()

    def get_new_file(self, index):
        if self.pgn_file != None:
            self.pgn_file.close()
        if ".bz2" in self.pgn_paths[index]:
            self.pgn_file = bz2.open(self.pgn_paths[index])
        elif "https://" in self.pgn_paths[index]:
            self.pgn_file = urlopen(self.pgn_paths[index])
        else:
            self.pgn_file = open(self.pgn_paths[index])

    def get_line(self):
        """
        get a new line with a new game from the currently used data file into the variable current_line.
        If all lines from the file have already been read,
        set the next file from the list pgn_paths to the pgn_file and read the first line
        :return: None
        """
        line = self.pgn_file.readline()
        if isinstance(line,bytes):
            line = line.decode("utf-8")
        if line == "":
            self.current_path_index += 1
            if self.current_path_index >= len(self.pgn_paths):
                self.current_path_index = 0
            self.get_new_file(self.current_path_index)
            line = self.pgn_file.readline()
        self.current_line = line[:-1].split()

    def get_position(self):
        """
        It makes the next move on the list and reads the positions on the board
        if it fails, I load another game from the data file and repeat the process
        :return: String of chessboard positions in FEN notation
        """
        if len(self.current_line) < 1:
            self.get_line()
            self.board.reset()
        try:
            self.board.push_san(self.current_line[0])
        except ValueError:
            self.get_line()
            self.board.reset()
            return self.get_position()
        self.current_line.pop(0)
        return self.board.fen()

    def play_func(self):
        """
        Main function of the generator, generates and pushes samples to the queue.
        :return: None, executes forever
        """
        while True:
            position = self.get_position()
            self.replay_queue.put(position)


class PGNIterator(object):
    def __init__(
        self,
        batch_size: int,
        generators_paths,
        replay_size=30000,
        replay_initial=10000,
        n_generators=16,
    ):
        """
        :param batch_size batch_size: size of batch yielded from replay buffer
        :param batch_size generators_paths: a list containing sets of paths for individual generators
        :param batch_size replay_size: size of replay buffer
        :param batch_size replay_initial: samples to initialize buffer before yielding
        :param batch_size n_generators: how many generators should generate data
        """
        if isinstance(generators_paths,str):
            n_generators = 1
        elif len(generators_paths) < n_generators:
            n_generators = len(generators_paths)
        self.replay_queue = mp.Queue(maxsize=n_generators * 4)
        self.generators = []
        for paths in generators_paths:
            self.generators.append(FENGenerator(self.replay_queue, paths))
        self.replay_buffer = deque(maxlen=replay_size)
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.replay_initial = replay_initial
        self.active_processes = []

    def kill_processes(self):
        """
        Stops executing all processes triggered by this class.
        :return: None
        """
        for process in self.active_processes:
            process.terminate()
            process.join()
        self.active_processes.clear()

    def initialize_processes(self, kill_old=True):
        """
        Starts executing processes that generate data, kills old processes if necessary.
        :param kill_old: it `True` then kills old processes, default: True
        :return: None
        """
        if kill_old:
            self.kill_processes()
        for generator in self.generators:
            process = mp.Process(target=generator.play_func, args=())
            process.start()
            self.active_processes.append(process)

    def close(self):
        for generator in self.generators:
            generator.closeFile()
        for process in self.active_processes:
            process.terminate()
            process.join()
        self.active_processes.clear()

    def __iter__(self):
        """
        Iterator function that samples batches from generated data and initializes processes.
        :return: yields batches of samples
        """
        try:
            self.initialize_processes()
            while True:
                while self.replay_queue.qsize():
                    entry = self.replay_queue.get()
                    self.replay_buffer.append(entry)
                if len(self.replay_buffer) < self.replay_initial:
                    continue
                yield random.sample(self.replay_buffer, self.batch_size)
        except Exception:
            pass
        finally:
            self.kill_processes()



class Preprocessor(object):
    def __init__(self, iterator: PGNIterator, device: torch.device):
        r"""
        Wrapper for `Iterator` class that preprocesses compressed data to pytorch tensors.
        :param iterator: iterator to wrap
        :param device: device for tensors
        """
        super(Preprocessor, self).__init__()
        self.iterator = iterator
        self.device = device

    def preprocess_fen(self, fen):
        r"""
        Preprocesses one fen and value,
        :param fen: tuple(fen, value), data to preprocess
        :return: dictionary of preprocessed tensors
        """

        raw_fen = fen
        fen = raw_fen.split('/')
        fen_dropped = fen[:-1]
        fen_dropped.extend(fen[-1].split())
        board_position_str = fen_dropped[:8]
        next_mover_str = fen_dropped[8]
        castling_str = fen_dropped[9]
        en_passant_str = fen_dropped[10]
        half_moves_str = fen_dropped[11]
        move_idx_str = fen_dropped[12]

        board_position = np.zeros((12, 8, 8,), dtype=np.float32)
        for r_idx, row in enumerate(board_position_str):
            column = 0
            for piece in row:
                try:
                    p = PIECES[piece]
                    board_position[p, column, r_idx] = 1.
                    column += 1
                except KeyError:
                    column += int(piece)

        next_mover = -1 if next_mover_str == 'b' else 1
        castling = np.zeros(4)
        for figure in castling_str:
            if figure == 'K':
                castling[0] = 1
            elif figure == 'Q':
                castling[1] = 1
            elif figure == 'k':
                castling[2] = 1
            elif figure == 'q':
                castling[3] = 1
        en_passant = np.zeros((1, 8, 8,), dtype=np.float32)
        if en_passant_str != '-':
            en_passant[0, FIRST_CORDS[en_passant_str[0]], SECOND_CORDS[en_passant_str[1]]] = 1.
        half_moves = int(half_moves_str)
        move_idx = int(move_idx_str)
        return {'board': torch.FloatTensor(board_position).to(self.device),
                'castling': torch.FloatTensor(castling).to(self.device),
                'en_passant': torch.FloatTensor(en_passant).to(self.device),

                'player': next_mover,
                'half_move': half_moves, 'full_move': move_idx,
                'fen': raw_fen}

    def preprocess_batch(self, batch):
        r"""
        Applies `preprocess_fen` for a batch of data.
        :param batch: batch to preprocess
        :return: dictionary of preprocessed tensors
        """
        boards = []
        players = []
        castlings = []
        en_passants = []
        half_moves = []
        full_moves = []
        fens = []
        for fen in batch:
            data = self.preprocess_fen(fen)
            boards.append(data['board'])
            players.append(data['player'])
            castlings.append(data['castling'])
            en_passants.append(data['en_passant'])
            half_moves.append(data['half_move'])
            full_moves.append(data['full_move'])
            fens.append(data['fen'])

        return {'boards': torch.stack(boards).to(self.device),
                'players': torch.FloatTensor(players).to(self.device),
                'castlings': torch.stack(castlings).to(self.device),
                'en_passants': torch.stack(en_passants).to(self.device),
                'half_moves': torch.FloatTensor(half_moves).to(self.device),
                'full_moves': torch.FloatTensor(full_moves).to(self.device)}

    def __iter__(self):
        r"""
        Like `Iterator.__iter__()`, but applies preprocessing to yielded data.
        :return: yields preprocessed batches of samples
        """
        for batch in self.iterator:
            yield self.preprocess_batch(batch)


class StandardConvSuite(object):
    def __init__(self, preprocessor):
        r"""
        Preprocesses dictionary of tensors to conv-suitable planes
        :param preprocessor: info source
        """
        self.preprocessor = preprocessor

    @staticmethod
    def preprocess_batch(batch):
        """
        Makes batch suitable for convolutional
        :param batch:
        :return:
        """
        boards = []
        for board, player in zip(batch['boards'], batch['players']):
            if player == 1:
                #swapping players' positions
                boards.append(board[[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]])
            else:
                boards.append(board)
        boards_v = torch.stack(boards)
        boards_v = torch.cat([boards_v, batch['en_passants']], dim=1)
        castlings = torch.zeros((boards_v.size()[0], 4, 8, 8)).to(boards_v.device)
        for idx, (castling, player) in enumerate(zip(batch['castlings'], batch['players'])):
            if player == 1:
                for c_idx, component in enumerate(castling[[2, 3, 0, 1]]):
                    castlings[idx, c_idx, :, :] = component
            else:
                for c_idx, component in enumerate(castling):
                    castlings[idx, c_idx, :, :] = component
        boards_v = torch.cat([boards_v, castlings], dim=1)
        return boards_v

    def __iter__(self):
        r"""
        Like `Preprocessor.__iter__()`, but applies preprocessing to yielded data.
        :return: yields preprocessed batches of samples
        """
        for batch in self.preprocessor:
            yield self.preprocess_batch(batch)


def getListOfDataset(path, n_generators =1, keyword=""):
    """
    Preprocesses dictionary of tensors to conv-suitable planes
    :param path: path to the dataset folder
    :param keyword: keyword appearing in each desired file
    :param n_generators : number of generators in the PGNIterator class
    :return: 2d list of paths
    """
    _, _, filenames = next(os.walk(path))
    filenames = [path+"//"+file for file in filenames if (keyword in file)]
    part = max(1,int(len(filenames)/n_generators ))
    if len(filenames) < 0:
        raise ValueError('there are no files with the given keyword')
    filenames = [filenames[i*part:]+filenames[:i*part] for i in range(n_generators )]
    return filenames
