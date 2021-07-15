import chess
from stockfish import Stockfish

import torch
import torch.multiprocessing as mp

from collections import deque
import numpy as np

import random
import os

import matplotlib.pyplot as plt


# initial setup that looks ugly
os.environ['OMP_NUM'] = '1'
first_cords = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
second_cords = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7}
pieces = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
          'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}


class Generator(object):
    def __init__(self, replay_queue: mp.Queue, stockfish_path: str,
                 *,
                 epsilon_start=1., epsilon_final=.0025, epsilon_frames=300000,
                 theta_start=1., theta_lifetime=.4, beta=.3, omega=.05, threads=2, speed=150):
        r"""
        Generates fen positions and evaluations by playing semi-stochastic games and pushes them to `replay queue`.

        :param replay_queue: target queue for samples
        :param epsilon_start: how many random moves to make after starting generation, default: 1.
        :param epsilon_final: how many random moves to make after epsilon decay process, default: .0025
        :param epsilon_frames: how much moves for epsilon decay process - at n move epsilon has value:
            max(epsilon_final, epsilon_start - n/epsilon_frames), default: 30000
        :param theta_start: how much random moves to make at the begging of each game (to use all openings), default: 1.
        :param theta_lifetime: how fast should theta decrease at n move of a game theta has value:
            theta_lifetime**n, default: .4
        :param beta: how often should play the weaker engine (2000 ELO instead of 3600), default: .3
        :param omega: how often should play the garbage engine (1300 ELO instead of 3600), default: .05
        :param threads: how many threads should the stockfish use, default: 2
        :param speed: how many milliseconds should stockfish think about its moves, when move is random,
            then stockfish doesn't think at all
        """
        super(Generator, self).__init__()

        self.eps_start = epsilon_start
        self.eps_final = epsilon_final
        self.eps_frames = epsilon_frames
        self.epsilon = epsilon_start
        self.frame = 0

        self.replay_queue = replay_queue

        self.theta_start = theta_start if theta_lifetime else 0.
        self.theta = theta_start
        self.theta_lifetime = theta_lifetime

        self.beta = beta
        self.omega = omega

        self.engine = Stockfish(
            path=stockfish_path,
            parameters={'Threads': threads})
        self.speed = speed

        self.moves = []

    def decay_epsilon(self):
        r"""
        One step of epsilon decay with respect to self.frame variable.
        :return: None
        """
        self.epsilon = max(self.eps_start - self.frame/self.eps_frames, self.eps_final)

    def engine_move(self):
        r"""
        Gets best move with stochasticity.
        :return: suggested move if any is possible, else `None`.
        """
        board = chess.Board(self.engine.get_fen_position())
        legal_moves = list(board.legal_moves)
        if not legal_moves or board.can_claim_threefold_repetition() or \
                board.can_claim_draw() or board.can_claim_fifty_moves():
            return None
        if np.random.rand() < self.theta + self.epsilon:
            return np.random.choice(legal_moves)
        return self.engine.get_best_move_time(self.speed)

    def get_move(self):
        r"""
        Makes move and resets board if necessary.
        :return: fen and evaluation of board after move
        """
        rand_state = np.random.rand()
        if rand_state < self.omega:
            self.engine.set_skill_level(4)  # 1200 ELO
        elif rand_state < self.beta:
            self.engine.set_skill_level(6)  # 2000 ELO
        else:
            self.engine.set_skill_level(20)  # 3600 ELO
        move = self.engine_move()
        if move is None:
            self.moves.clear()
            self.engine.set_position([])
            self.theta = self.theta_start
            move = self.engine_move()
        self.moves.append(move)
        self.engine.set_position(self.moves)
        self.theta *= self.theta_lifetime
        fen = self.engine.get_fen_position()
        evaluation = self.engine.get_evaluation()
        if evaluation['type'] == 'mate':
            return fen, 20.
        else:
            return fen, evaluation['value'] / 500.

    def play_func(self):
        r"""
        Main function of the generator, generates and pushes samples to the queue.
        :return: None, executes forever
        """
        while True:
            position = self.get_move()
            self.replay_queue.put(position)
            self.frame += 1
            self.decay_epsilon()


class Iterator(object):
    def __init__(self, batch_size: int, stockfish_path: str, replay_size=30000, replay_initial=10000, n_generators=16,
                 eps_final_mult=(0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 2., 2., 16., 16., 16., 16.),
                 theta_lifetime_mult=(0., .25, 1., 2., 0., .25, 1., 2., 0., .25, 1., 2., 0., .25, 1., 2.),
                 beta_mult=(0., .5, 1., 16., 0., .5, 1., 16., 0., .5, 1., 16., 0., .5, 1., 16.),
                 omega_mult=(0., .5, 2., 16., 0., .5, 2., 16., 0., .5, 2., 16., 0., .5, 2., 16.),
                 *,
                 epsilon_start=1., epsilon_final=.001, epsilon_frames=300000,
                 theta_start=1., theta_lifetime=.4, beta=.3, omega=.05, threads=2, speed=150):
        r"""
        Iterable class that generates stochastic fen and value samples by parallel running generators.
        :param batch_size: size of batch yielded from replay buffer
        :param replay_size: size of replay buffer
        :param replay_initial: samples to initialize buffer before yielding
        :param n_generators: how many generators should generate data
        :param eps_final_mult: allows to specify unique `epsilon_final` for each generator
        :param theta_lifetime_mult: allows to specify unique `theta_lifetime` for each generator
        :param beta_mult: allows to specify unique `beta` for each generator
        :param omega_mult: allows to specify unique `omega` for each generator
        :param epsilon_start, epsilon_final, epsilon_frames, theta_start, theta_lifetime,
               beta, omega, threads, speed: all arguments for generators, check docstring
               of the Generator class
        """
        super(Iterator, self).__init__()
        if len(eps_final_mult) < n_generators:
            eps_final_mult = eps_final_mult * (n_generators // len(eps_final_mult) + 1)
        if len(theta_lifetime_mult) < n_generators:
            theta_lifetime_mult = theta_lifetime_mult * (n_generators // len(theta_lifetime_mult) + 1)
        if len(beta_mult) < n_generators:
            beta_mult = beta_mult * (n_generators // len(beta_mult) + 1)
        if len(omega_mult) < n_generators:
            omega_mult = omega_mult * (n_generators // len(omega_mult) + 1)
        eps_final_mult = eps_final_mult[:n_generators]
        theta_lifetime_mult = theta_lifetime_mult[:n_generators]
        beta_mult = beta_mult[:n_generators]
        omega_mult = omega_mult[:n_generators]

        self.replay_queue = mp.Queue(maxsize=n_generators * 4)
        self.generators = []
        for e_final, t_life, b, o in zip(eps_final_mult, theta_lifetime_mult, beta_mult, omega_mult):
            self.generators.append(Generator(self.replay_queue, stockfish_path,
                                             epsilon_start=epsilon_start, epsilon_final=epsilon_final*e_final,
                                             epsilon_frames=epsilon_frames, theta_start=theta_start,
                                             theta_lifetime=theta_lifetime*t_life, beta=beta*b, omega=omega*o,
                                             threads=threads, speed=speed))
        self.replay_buffer = deque(maxlen=replay_size)
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.replay_initial = replay_initial
        self.active_processes = []

    def kill_processes(self):
        r"""
        Stops executing all processes triggered by this class.
        :return: None
        """
        for process in self.active_processes:
            process.terminate()
            process.join()
        self.active_processes.clear()

    def initialize_processes(self, kill_old=True):
        r"""
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
        r"""
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
    def __init__(self, iterator: Iterator, device: torch.device):
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

        raw_fen, value = fen
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
                    p = pieces[piece]
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
            en_passant[0, first_cords[en_passant_str[0]], second_cords[en_passant_str[1]]] = 1.
        half_moves = int(half_moves_str)
        move_idx = int(move_idx_str)
        return {'board': torch.FloatTensor(board_position).to(self.device),
                'castling': torch.FloatTensor(castling).to(self.device),
                'en_passant': torch.FloatTensor(en_passant).to(self.device),

                'player': next_mover, 'q_value': value,
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
        q_values = []
        half_moves = []
        full_moves = []
        fens = []
        for fen in batch:
            data = self.preprocess_fen(fen)
            boards.append(data['board'])
            players.append(data['player'])
            castlings.append(data['castling'])
            en_passants.append(data['en_passant'])
            q_values.append(data['q_value'])
            half_moves.append(data['half_move'])
            full_moves.append(data['full_move'])
            fens.append(data['fen'])

        return {'boards': torch.stack(boards).to(self.device),
                'players': torch.FloatTensor(players).to(self.device),
                'castlings': torch.stack(castlings).to(self.device),
                'en_passants': torch.stack(en_passants).to(self.device),
                'q_values': torch.FloatTensor(q_values).to(self.device),
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


# DEMO
if __name__ == '__main__':
    plt.ion()
    plt.show()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    for X_batch_v in StandardConvSuite(Preprocessor(
            Iterator(64, 'stockfish_13_linux_x64_avx2'),
            torch.device('cuda'))):
        for sample in X_batch_v:
            ax2.set_title(f'min: {X_batch_v.min()}'
                          f'max: {X_batch_v.max()}'
                          f'mean: {X_batch_v.mean()}'
                          f'size: {X_batch_v.size()}')
            board_viz = (np.arange(1, 13)[:, None, None] * sample[:12].data.cpu().numpy()).mean(axis=0)
            en_passant_viz = sample[12].data.cpu().numpy()
            castlings_viz = sample[13:].data.cpu().numpy().mean(axis=0)
            ax1.imshow(board_viz, cmap='jet')
            ax2.imshow(en_passant_viz, vmin=0, vmax=1, cmap='jet')
            ax3.imshow(castlings_viz, vmin=0, vmax=1, cmap='jet')
            plt.draw()
            plt.pause(.01)
