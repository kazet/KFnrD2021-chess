import chess
import torch.multiprocessing as mp
import bz2

class Generator_pgn():
    def __init__(self,replay_queue: mp.Queue,pgn_path):
        
        r"""
        Iterable class that generates stochastic fen and value samples by parallel running generators.
        :param replay_queue: target queue for samples
        :param current_line: currently used line from the data file, representing one game
        :param board: the chessboard on which the moves are made
        :param pgn_paths: list of paths to data files
        :param current_path: index of the path to the currently used data file in the pgn_paths list
        :param pgn_file: currently used data file
        """
        
        self.replay_queue=replay_queue;
        self.current_path=0
        self.current_line=[]
        self.board=chess.Board()
        if(type(pgn_path)==str):    
            self.pgn_paths=[pgn_path]
        else:
            self.pgn_paths=pgn_path
        if(".bz2" in self.pgn_paths[0]):
            self.pgn_file=bz2.open(self.pgn_paths[0])
        else:
            self.pgn_file=open(self.pgn_paths[0])
            
        
    def get_line(self):
        r"""
        get a new line with a new game from the currently used data file into the variable current_line.
        If all lines from the file have already been read, set the next file from the list pgn_paths to the pgn_file and read the first line
        :return: None
        """
        line=self.pgn_file.readline()
        if(type(line)==bytes):
                line=line.decode("utf-8")
        if(line==""):
            self.current_path=(self.current_path+1)%len(self.pgn_paths)
            if(".bz2" in self.pgn_paths[self.current_path]):
                self.pgn_file=bz2.open(self.pgn_paths[self.current_path])
            else:
                self.pgn_file=open(self.pgn_paths[self.current_path])
            line=self.pgn_file.readline()
        self.current_line=line[:-1].split()
    def get_move(self):
        r"""
        It makes the next move on the list and reads the positions on the board
        if it fails, I load another game from the data file and repeat the process
        :return: String of chessboard positions in FEN notation
        """
        if(len(self.current_line)<1):
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
            return self.get_move()
            
    def play_func(self):
        r"""
        Main function of the generator, generates and pushes samples to the queue.
        :return: None, executes forever
        """
        while True:
            position = self.get_move()
            self.replay_queue.put(position)
        

#Test
replay_queue=mp.Queue(20)      
obj=Generator_pgn(replay_queue,["lichessE.pgn",'lichess.pgn.bz2'])
for i in range(1000):
    print(obj.get_move())
