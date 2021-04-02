import chess
import torch.multiprocessing as mp
import bz2
from collections import deque
import random


class GeneratorPgn():
    def __init__(self, replay_queue: mp.Queue, pgn_path):

        """
        Iterable class that generates stochastic fen and value samples by parallel running generators.
        :param replay_queue: target queue for samples
        :param current_line: currently used line from the data file, representing one game
        :param board: the chessboard on which the moves are made
        :param pgn_paths: list of paths to data files
        :param current_path: index of the path to the currently used data file in the pgn_paths list
        :param pgn_file: currently used data file
        """

        self.replay_queue = replay_queue
        self.current_path = 0
        self.current_line = []
        self.board = chess.Board()
        if type(pgn_path) == str:
            self.pgn_paths = [pgn_path]
        else:
            self.pgn_paths = pgn_path
        if ".bz2" in self.pgn_paths[0]:
            self.pgn_file = bz2.open(self.pgn_paths[0])
        else:
            self.pgn_file = open(self.pgn_paths[0])

    def get_line(self):
        """
        get a new line with a new game from the currently used data file into the variable current_line.
        If all lines from the file have already been read, set the next file from the list pgn_paths to the pgn_file and read the first line
        :return: None
        """
        line = self.pgn_file.readline()
        if type(line) == bytes:
            line = line.decode("utf-8")
        if line == "":
            self.current_path += 1
            if self.current_path >= len(self.pgn_paths):
                self.current_path = 0
            if self.pgn_paths[self.current_path].endswith(".bz2"):
                self.pgn_file = bz2.open(self.pgn_paths[self.current_path])
            else:
                self.pgn_file = open(self.pgn_paths[self.current_path])
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
            self.current_line.pop(0)
            return self.board.fen()
        except:
            print(self.current_line[0])
            self.get_line()
            self.board.reset()
            return self.get_position()

    def play_func(self):
        """
        Main function of the generator, generates and pushes samples to the queue.
        :return: None, executes forever
        """
        while True:
            position = self.get_position()
            self.replay_queue.put(position)


class IteratorPgn(object):
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
        if type(generators_paths) == str:
            n_generators = 1
        elif len(generators_paths) < n_generators:
            n_generators = len(generators_paths)
        self.replay_queue = mp.Queue(maxsize=n_generators * 4)
        self.generators = []
        for paths in generators_paths:
            self.generators.append(GeneratorPgn(self.replay_queue, paths))
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
                print("koko9")
                print(random.sample(self.replay_buffer, self.batch_size), "kkk")
                print("jij")
                yield random.sample(self.replay_buffer, self.batch_size)
        except Exception:
            pass
        finally:
            self.kill_processes()
