from tkinter import *
from PIL import ImageTk, Image
from functools import partial
import os
import chess
from tkinter import filedialog
import torch

from inference import Inference
import settings
from model import Coder
from find_similar import find_similar,similarity_functions


class Main(Frame):
    def __init__(
        self,
        master,
        color_palette,
        home_path,
        pieces_name,
        pieces_path,
        arrows_path,
        images_sizes,
        selected_piece,
        pieces_padding,
        header_height = 60,
        option_width = 100,
        message_width = 600,
        message_height = 200,
        find_options_width = 600,
        find_options_height = 400,
    ):
        Frame.__init__(self, master, bg=color_palette[0])
        master.rowconfigure(0, weight=1)
        master.columnconfigure(0, weight=1)
        self.main = self
        self.coder_set = False
        self.home_path = home_path
        self.current_pages = 0
        self.pages_fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
        self.pieces_name = pieces_name
        self.pieces_path = pieces_path
        self.arrows_path = arrows_path
        self.images_sizes = images_sizes
        self.selected_piece = selected_piece
        self.pieces_padding = pieces_padding
        self.color_palette = color_palette
        self.entering = True
        self.fen_placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        self.fen_player = "w"
        self.fen_castling = "KQkq"
        self.header_height = header_height
        self.option_width = option_width
        self._create_widgets(message_width, message_height, find_options_width, find_options_height)
        self.bind("<Configure>", self._resize)
        self.winfo_toplevel().minsize(600, 600)
        self.display_fen()
        self.coder = None
        self.games = None
        self.coder_launcher = None
        self.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def _create_widgets(self, message_width, message_height,find_options_width, find_options_height):
        self.board_box = BoardBox(self)
        self.option_box = Options(self, self.option_width)
        self.header = Header(self, header_height=self.header_height)
        self.pgn_box = PGNOptions(self, self.option_width)
        self.tensor_message = TensorDisplayer(self, message_width, message_height)
        self.find_option = FindOptions(self,find_options_width, find_options_height)

        self.board_box.grid(row=1, column=0, sticky=N + S + E + W)
        self.option_box.grid(row=1, column=1, sticky=N + S + E + W)
        self.header.grid(row=0, column=0, columnspan=2, sticky=N + S + E + W)

        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

    def _resize(self, event):
        """Modify padding when window is resized."""
        w, h = event.width, event.height
        self.rowconfigure(1, weight=h - self.header_height)
        self.columnconfigure(0, weight=w - self.option_width)

    def display_fen(self):
        self.header.display_fen(self.fen_placement, self.fen_player, self.fen_castling)
        if self.entering:
            self.pages_fens[self.current_pages] = " ".join(
                [self.fen_placement, self.fen_player, self.fen_castling, "- 0 0"]
            )

    def set_fen(self, fen=None):
        if self.entering == False:
            fen = self.games.board.fen()
            split_fen = fen.split()
            self.fen_placement = split_fen[0]
            self.fen_player = split_fen[1]
            self.fen_castling = split_fen[2]
            self.board_box.board.set_board(self.fen_placement)
            return
        try:
            a = chess.Board(fen)
            fen = a.fen()
            del a
            split_fen = fen.split()
            self.fen_placement = split_fen[0]
            self.fen_player = split_fen[1]
            self.fen_castling = split_fen[2]
            self.option_box.set_option(self.fen_player, self.fen_castling)
            self.board_box.board.set_board(self.fen_placement)
            self.pages_fens[self.current_pages] = fen
        except ValueError:
            self.header.display_fen("Incorrect fen", "", "")

    def set_coder(self, filename):
        try:
            self.coder = Coder(settings.BOARD_SHAPE, settings.LATENT_SIZE).to(
                settings.DEVICE
            )
            self.coder.load_state_dict(torch.load(filename))
            self.coder.eval()
            self.coder_launcher = Inference(
                settings.DEVICE,
                self.coder,
            )
            self.coder_set = True
            return True
        except Exception:
            return False

    def show_find_option(self):
        if self.coder_set:
            self.find_option.place(relx=0.5, rely=0.5, anchor=CENTER)
        else:
            self.header.coder_label["text"]="Set Coder first"

    def run_coder(self,number,comparison):
        if self.coder_set:
            output = str(
                self.coder_launcher.predict([self.pages_fens[self.current_pages]])
            )
            self.find_option.place_forget()
            self.display_tensor(output)
            self.games = find_similar(self.pages_fens[self.current_pages], number, similarity_functions[comparison])
            self.entering = False
            self.set_fen()
            self.option_box.grid_forget()
            self.pgn_box.grid(row=1, column=1, sticky=N + S + E + W)
            self.pgn_box.set_info(self.games.get_info())
            self.pgn_box.set_position_number()

    def games_command(self):
        self.games.last_move()
        self.set_fen()

    def exit_coder(self):
        self.pgn_box.grid_forget()
        self.option_box.grid(row=1, column=1, sticky=N + S + E + W)
        self.entering = True
        self.set_fen(self.pages_fens[self.current_pages])

    def display_tensor(self, message):
        self.tensor_message.set_message(message)
        self.tensor_message.place(relx=0.5, rely=0.5, anchor=CENTER)

    def stop_display_tensor(self):
        self.tensor_message.place_forget()


class BoardBox(Frame):
    def __init__(self, master):
        Frame.__init__(self, master, bg=master.color_palette[4])
        master.rowconfigure(0, weight=1)
        master.columnconfigure(0, weight=1)
        self.main = master.main
        self._create_widgets()
        self.bind("<Configure>", self._resize)
        self.winfo_toplevel().minsize(150, 150)

    def _create_widgets(self):
        self.board = Board(self)
        self.board.grid(row=1, column=1, sticky=N + S + E + W)

        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

    def _resize(self, event):
        """Modify padding when window is resized."""
        w, h = event.width, event.height
        if w > h:
            self.rowconfigure(0, weight=0)
            self.rowconfigure(1, weight=1)
            self.rowconfigure(2, weight=0)
            self.columnconfigure(0, weight=int((w - h) / 2))
            self.columnconfigure(1, weight=h)
            self.columnconfigure(2, weight=int((w - h) / 2))
        elif w < h:
            self.rowconfigure(0, weight=int((h - w) / 2))
            self.rowconfigure(1, weight=w)
            self.rowconfigure(2, weight=int((h - w) / 2))
            self.columnconfigure(0, weight=0)
            self.columnconfigure(1, weight=1)
            self.columnconfigure(2, weight=0)
        else:
            self.rowconfigure(0, weight=0)
            self.rowconfigure(1, weight=w)
            self.rowconfigure(2, weight=0)
            self.columnconfigure(0, weight=0)
            self.columnconfigure(1, weight=1)
            self.columnconfigure(2, weight=0)


class Board(Frame):
    def __init__(self, master, border_proportion=0.02):
        self.main = master.main
        Frame.__init__(self, master, bg=self.main.color_palette[0])
        master.rowconfigure(0, weight=1)
        master.columnconfigure(0, weight=1)
        self.border_proportion = border_proportion
        self._create_widgets()
        self.bind("<Configure>", self._resize)

    def _create_widgets(self):
        self.content = []
        self.board = []
        self.border = Frame(self)
        self.border.place(
            relx=self.border_proportion,
            rely=self.border_proportion,
            relwidth=1 - 2 * self.border_proportion,
            relheight=1 - 2 * self.border_proportion,
        )
        for x in range(8):
            self.content.append([])
            self.board.append([])
            for y in range(8):
                color = self.main.color_palette[9]
                if (x + y) % 2 == 0:
                    color = self.main.color_palette[2]
                self.board[x].append("")
                command = partial(self.replace, x, y)
                self.content[x].append(
                    Button(
                        self.border,
                        relief=FLAT,
                        bg=color,
                        activebackground=color,
                        command=command,
                        image=None,
                    )
                )
                self.content[x][y].place(
                    relx=0.125 * x, rely=0.125 * y, relwidth=0.125, relheight=0.125
                )

    def _resize(self, event):
        for x in range(8):
            for y in range(8):
                if self.board[x][y] != "":
                    image = self.get_image(self.board[x][y])
                    self.content[x][y].config(image=image)
                    self.content[x][y].image = image

    def replace(self, x, y):
        if self.main.entering:
            if (
                self.board[x][y] == self.main.selected_piece
                or self.main.selected_piece == ""
            ) and self.board[x][y] != "":
                self.content[x][y].config(image="")
                self.board[x][y] = ""
            elif self.main.selected_piece != "":
                self.board[x][y] = self.main.selected_piece
                image = self.get_image(self.main.selected_piece)
                self.content[x][y].config(image=image)
                self.content[x][y].image = image
            self.main.fen_placement = self.get_fen()
            self.main.display_fen()

    def get_size(self, num):
        if num <= self.main.images_sizes[0] + self.main.pieces_padding:
            return self.main.images_sizes[0]
        else:
            for i in self.main.images_sizes[1:]:
                if num < i + self.main.pieces_padding:
                    return i
        return self.main.images_sizes[-1]

    def get_image(self, piece):
        size = self.get_size(
            (self.winfo_width() * (1 - 2 * self.border_proportion)) / 8
        )
        image = Image.open(
            os.path.join(
                self.main.home_path,
                "img",
                str(size)
                + self.main.pieces_name
                + self.main.pieces_path[piece]
                + ".png",
            )
        )
        return ImageTk.PhotoImage(image)

    def get_fen(self):
        fen = ""
        space = 0
        for y in range(8):
            space = 0
            for x in range(8):
                if self.board[x][y] == "":
                    space += 1
                    continue
                if space > 0:
                    fen += str(space)
                fen += self.board[x][y]
                space = 0
            if space > 0:
                fen += str(space)
            fen += "/"
        return fen[:-1]

    def set_board(self, fen):
        for y, i in enumerate(fen.split("/")):
            x = 0
            for j in i:
                if j.isnumeric():
                    for k in range(int(j)):
                        self.content[x][y].config(image="")
                        self.board[x][y] = ""
                        x += 1
                else:
                    self.board[x][y] = j
                    image = self.get_image(j)
                    self.content[x][y].config(image=image)
                    self.content[x][y].image = image
                    x += 1


class Options(Frame):
    def __init__(self, master, option_width):
        self.main = master.main
        Frame.__init__(self, master, bg=self.main.color_palette[5], width=option_width)
        master.rowconfigure(0, weight=1)
        master.columnconfigure(0, weight=1)
        self._create_widgets(option_width)
        self.last_selected_button = (0, 0)

    def _create_widgets(self, option_width):
        pointer_y = 0
        self.piece_label = Label(
            self,
            text="PIECES",
            bg=self.main.color_palette[2],
            fg=self.main.color_palette[0],
        )
        self.piece_box = Frame(self, width=option_width, height=option_width * 3)
        self.castling_label = Label(
            self,
            text="CASTLING",
            bg=self.main.color_palette[2],
            fg=self.main.color_palette[0],
        )
        self.castling_box = Frame(self, width=option_width, height=option_width)
        self.player_label = Label(
            self,
            text="PLAYER\nWHO MOVES",
            bg=self.main.color_palette[2],
            fg=self.main.color_palette[0],
        )
        self.player_box = Frame(self, width=option_width, height=int(option_width / 2))

        self.piece_label.grid(row=0, column=0, sticky=E + W, pady=2)
        self.piece_box.grid(row=1, column=0, sticky=E + W, ipadx=2, padx=2)
        self.castling_label.grid(row=2, column=0, sticky=E + W, pady=2)
        self.castling_box.grid(row=3, column=0, sticky=E + W, ipadx=2, padx=2)
        self.player_label.grid(row=4, column=0, sticky=E + W, pady=2)
        self.player_box.grid(row=5, column=0, sticky=E + W, ipadx=2, padx=2)

        self.piece_buttons = []
        pieces = [["K", "Q", "R", "N", "B", "P"], ["k", "q", "r", "n", "b", "p"]]
        for column in range(2):
            self.piece_buttons.append([])
            for row in range(6):
                size = self.get_size(option_width / 2)
                image = Image.open(
                    os.path.join(
                        self.main.home_path,
                        "img",
                        str(size)
                        + self.main.pieces_name
                        + self.main.pieces_path[pieces[column][row]]
                        + ".png",
                    )
                )
                image = ImageTk.PhotoImage(image)
                command = partial(self.select, column, row, pieces[column][row])
                self.piece_buttons[column].append(
                    Button(
                        self.piece_box,
                        bg=self.main.color_palette[7],
                        activebackground=self.main.color_palette[8],
                        command=command,
                    )
                )
                self.piece_buttons[column][row].config(image=image)
                self.piece_buttons[column][row].image = image
                self.piece_buttons[column][row].place(
                    relx=0.5 * column, rely=row / 6, relwidth=0.5, relheight=1 / 6
                )

        self.castling = [0, 0, 0, 0]
        size = self.get_size(option_width / 2 - 10)
        image = [
            ImageTk.PhotoImage(
                Image.open(
                    os.path.join(
                        self.main.home_path,
                        "img",
                        str(size) + self.main.arrows_path[i],
                    )
                )
            )
            for i in range(4)
        ]
        self.castling_button = []
        self.castling_button.append(
            Button(
                self.castling_box,
                image=image[0],
                bg=self.main.color_palette[7],
                activebackground=self.main.color_palette[8],
                command=partial(self.display_castling, 0),
            )
        )
        self.castling_button.append(
            Button(
                self.castling_box,
                image=image[1],
                bg=self.main.color_palette[7],
                activebackground=self.main.color_palette[8],
                command=partial(self.display_castling, 1),
            )
        )
        self.castling_button.append(
            Button(
                self.castling_box,
                image=image[2],
                bg=self.main.color_palette[7],
                activebackground=self.main.color_palette[8],
                command=partial(self.display_castling, 2),
            )
        )
        self.castling_button.append(
            Button(
                self.castling_box,
                image=image[3],
                bg=self.main.color_palette[7],
                activebackground=self.main.color_palette[8],
                command=partial(self.display_castling, 3),
            )
        )

        for i in range(4):
            self.castling_button[i].image = image[i]

        self.castling_button[1].place(relx=0, rely=0.5, relwidth=0.5, relheight=0.5)
        self.castling_button[3].place(relx=0, rely=0, relwidth=0.5, relheight=0.5)
        self.castling_button[0].place(relx=0.5, rely=0.5, relwidth=0.5, relheight=0.5)
        self.castling_button[2].place(relx=0.5, rely=0, relwidth=0.5, relheight=0.5)

        size = self.get_size(option_width / 2)
        image = Image.open(
            os.path.join(
                self.main.home_path,
                "img",
                str(size) + self.main.pieces_name + self.main.pieces_path["K"] + ".png",
            )
        )
        image = ImageTk.PhotoImage(image)
        self.player_w = Button(
            self.player_box,
            bg=self.main.color_palette[3],
            activebackground=self.main.color_palette[8],
            command=partial(self.display_player, "w"),
            image=image,
        )
        self.player_w.image = image
        image = Image.open(
            os.path.join(
                self.main.home_path,
                "img",
                str(size) + self.main.pieces_name + self.main.pieces_path["k"] + ".png",
            )
        )
        image = ImageTk.PhotoImage(image)
        self.player_b = Button(
            self.player_box,
            bg=self.main.color_palette[7],
            activebackground=self.main.color_palette[8],
            command=partial(self.display_player, "b"),
            image=image,
        )
        self.player_b.image = image

        self.player_w.place(relx=0, rely=0, relwidth=0.5, relheight=1)
        self.player_b.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

    def select(self, column, row, piece):
        if self.main.selected_piece == piece:
            self.piece_buttons[column][row].config(bg=self.main.color_palette[7])
            self.main.selected_piece = ""
        else:
            self.piece_buttons[self.last_selected_button[0]][
                self.last_selected_button[1]
            ].config(bg=self.main.color_palette[7])
            self.last_selected_button = (column, row)
            self.piece_buttons[column][row].config(bg=self.main.color_palette[3])
            self.main.selected_piece = piece

    def get_size(self, num):
        if num <= self.main.images_sizes[0] + self.main.pieces_padding:
            return self.main.images_sizes[0]
        else:
            for i in self.main.images_sizes[1:]:
                if num < i + self.main.pieces_padding:
                    return i
        return self.main.images_sizes[-1]

    def display_castling(self, castling_type):
        if self.castling[castling_type] == 0:
            self.castling[castling_type] = 1
            self.castling_button[castling_type].config(bg=self.main.color_palette[3])
        else:
            self.castling[castling_type] = 0
            self.castling_button[castling_type].config(bg=self.main.color_palette[7])
        if 1 in self.castling:
            fen = ""
            castling_order = ["K", "Q", "k", "q"]
            for i in range(4):
                if self.castling[i] == 1:
                    fen += castling_order[i]
            self.main.fen_castling = fen
        else:
            self.main.fen_castling = "-"
        self.main.display_fen()

    def display_player(self, player):
        if player == "w":
            self.player_w.config(bg=self.main.color_palette[3])
            self.player_b.config(bg=self.main.color_palette[7])
            self.main.fen_player = "w"
        else:
            self.player_b.config(bg=self.main.color_palette[3])
            self.player_w.config(bg=self.main.color_palette[7])
            self.main.fen_player = "b"
        self.main.display_fen()

    def set_option(self, player, castling):
        self.display_player(player)
        castling_order = ["K", "Q", "k", "q"]
        for i, c in enumerate(castling_order):
            if c in castling:
                self.castling_button[i].config(bg=self.main.color_palette[3])
                self.castling[i] = 1
            else:
                self.castling[i] = 0
                self.castling_button[i].config(bg=self.main.color_palette[7])


class Header(Frame):
    def __init__(self, master, header_height):
        self.main = master.main
        Frame.__init__(self, master, bg=self.main.color_palette[5])
        master.rowconfigure(0, weight=1)
        master.columnconfigure(0, weight=1)
        self._create_widgets(header_height)
        self.last_selected_button = (0, 0)
        self.bind("<Configure>", self._resize)

    def _create_widgets(self, header_height):
        self.fen_box = Frame(self, bg=self.main.color_palette[9])
        self.fen_entry = Entry(
            self.fen_box,
            justify=CENTER,
            bg=self.main.color_palette[2],
            fg=self.main.color_palette[0],
            font=("Courier", 12),
        )
        self.set_fen = Button(
            self.fen_box,
            text="SET FEN",
            command=self.main_set_fen,
            bg=self.main.color_palette[2],
            fg=self.main.color_palette[0],
        )
        self.fen_entry.bind("<Return>")
        self.coder_box = Frame(self, bg=self.main.color_palette[9])
        self.coder_run = Button(
            self.coder_box,
            text="FIND SIMILAR POSITION",
            command=self.main.show_find_option,
            bg=self.main.color_palette[2],
            fg=self.main.color_palette[0],
        )
        self.coder_path = Button(
            self.coder_box,
            text="SET CODER",
            command=self.get_coder,
            bg=self.main.color_palette[2],
            fg=self.main.color_palette[0],
        )
        self.coder_label = Label(
            self.coder_box,
            text="Coder not set",
            bg=self.main.color_palette[2],
            fg=self.main.color_palette[0],
        )

        self.fen_box.grid(
            row=0, column=1, sticky=E + W, padx=2, pady=2, ipadx=2, ipady=2
        )
        self.coder_box.grid(
            row=0, column=0, sticky=E + W, padx=2, pady=2, ipadx=2, ipady=2
        )

        self.set_fen.grid(row=0, column=0, sticky=E + W, pady=(4, 0), padx=(4, 0))
        self.fen_entry.grid(row=0, column=1, sticky=N + S + E + W, pady=(3, 0), padx=4)

        self.coder_path.grid(row=0, column=0, sticky=E + W, pady=(4, 0), padx=4)
        self.coder_label.grid(row=0, column=1, sticky=E + W, pady=(4, 0), padx=(0, 4))
        self.coder_run.grid(row=0, column=2, sticky=E + W, pady=(4, 0))

    def _resize(self, event):
        """Modify padding when window is resized."""
        w, h = event.width, event.height
        self.columnconfigure(1, weight=int(w))
        self.fen_box.columnconfigure(1, weight=int(w))

    def display_fen(self, fen_placement, fen_player, fen_castling):
        self.fen_entry.delete(0, END)
        self.fen_entry.insert(0, " ".join([fen_placement, fen_player, fen_castling]))

    def main_set_fen(self, event=None):
        if self.main.entering:
            self.main.set_fen(self.fen_entry.get())

    def get_coder(self):
        filename = filedialog.askopenfilename(
            filetypes=[("model file", ("*.pt", "*.pth"))]
        )
        if self.main.set_coder(filename):
            self.coder_label["text"] = "Coder set"
        else:
            self.coder_label["text"] = "Coder couldn't be set"


class PGNOptions(Frame):
    def __init__(self, master, option_width):
        self.main = master.main
        Frame.__init__(self, master, bg=self.main.color_palette[5], width=option_width)
        self.max_length = int(option_width/6.5)
        master.rowconfigure(0, weight=1)
        master.columnconfigure(0, weight=1)
        self._create_widgets(option_width)
        self.last_selected_button = (0, 0)

    def _create_widgets(self, option_width):
        pointer_y = 0
        self.exit_button = Button(
            self,
            text="BACK",
            bg=self.main.color_palette[3],
            activebackground=self.main.color_palette[4],
            command=self.main.exit_coder,
        )
        self.info_box = Frame(
            self,
            bg=self.main.color_palette[2],
        )
        self.info_labels = []
        self.game_change = Label(
            self,
            text="CHANGE\nGAME",
            bg=self.main.color_palette[2],
            fg=self.main.color_palette[0],
        )
        self.game_box = Frame(self, width=option_width, height=int(option_width / 3))
        self.position_change = Label(
            self,
            text="CHANGE\nMOVE",
            bg=self.main.color_palette[2],
            fg=self.main.color_palette[0],
        )
        self.position_box = Frame(
            self, width=option_width, height=int(option_width / 3)
        )

        self.exit_button.grid(row=0, column=0, sticky=E + W, pady=2)
        self.game_change.grid(row=1, column=0, sticky=E + W, pady=2)
        self.game_box.grid(row=2, column=0, sticky=E + W, ipadx=2, padx=2)
        self.position_change.grid(row=3, column=0, sticky=E + W, pady=2)
        self.position_box.grid(row=4, column=0, sticky=E + W, ipadx=2, padx=2)
        self.info_box.grid(row=5, column=0, sticky=E + W, pady=2)

        self.castling = [0, 0, 0, 0]
        size = self.get_size(option_width / 3 - 5)
        image = [
            ImageTk.PhotoImage(
                Image.open(
                    os.path.join(
                        self.main.home_path,
                        "img",
                        str(size) + self.main.arrows_path[i],
                    )
                )
            )
            for i in range(2, 4)
        ]
        self.last_game = Button(
            self.game_box,
            image=image[1],
            bg=self.main.color_palette[7],
            activebackground=self.main.color_palette[8],
            command=lambda: [
                self.main.games.last_game(),
                self.main.set_fen(),
                self.set_game_number(),
            ],
        )
        self.last_game.image = image[1]

        self.next_game = Button(
            self.game_box,
            image=image[0],
            bg=self.main.color_palette[7],
            activebackground=self.main.color_palette[8],
            command=lambda: [
                self.main.games.next_game(),
                self.main.set_fen(),
                self.set_game_number(),
            ],
        )
        self.next_game.image = image[0]

        self.game_number = Label(
            self.game_box,
            text="1",
            bg=self.main.color_palette[8],
            fg=self.main.color_palette[2],
            font=("TkDefaultFont", 16),
        )

        self.last_game.place(relx=0, rely=0, relwidth=1 / 3, relheight=1)
        self.next_game.place(relx=2 / 3, rely=0, relwidth=1 / 3, relheight=1)
        self.game_number.place(relx=1 / 3, rely=0, relwidth=1 / 3, relheight=1)

        self.last_position = Button(
            self.position_box,
            image=image[1],
            bg=self.main.color_palette[7],
            activebackground=self.main.color_palette[8],
            command=lambda: [
                self.main.games.last_move(),
                self.main.set_fen(),
                self.set_position_number(),
            ],
        )
        self.last_position.image = image[1]

        self.next_position = Button(
            self.position_box,
            image=image[0],
            bg=self.main.color_palette[7],
            activebackground=self.main.color_palette[8],
            command=lambda: [
                self.main.games.next_move(),
                self.main.set_fen(),
                self.set_position_number(),
            ],
        )
        self.next_position.image = image[0]

        self.position_number = Label(
            self.position_box,
            text="1",
            bg=self.main.color_palette[8],
            fg=self.main.color_palette[2],
            font=("TkDefaultFont", 16),
        )

        self.last_position.place(relx=0, rely=0, relwidth=1 / 3, relheight=1)
        self.next_position.place(relx=2 / 3, rely=0, relwidth=1 / 3, relheight=1)
        self.position_number.place(relx=1 / 3, rely=0, relwidth=1 / 3, relheight=1)

    def get_size(self, num):
        if num <= self.main.images_sizes[0] + self.main.pieces_padding:
            return self.main.images_sizes[0]
        else:
            for i in self.main.images_sizes[1:]:
                if num < i + self.main.pieces_padding:
                    return i
        return self.main.images_sizes[-1]

    def set_info(self, info):
        text = ""
        for label in self.info_labels:
            label.grid_forget()
        del self.info_labels
        self.info_labels = []
        for i in info:
            if info[i] != "":
                self.info_labels.append(
                    Label(
                        self.info_box,
                        bg=self.main.color_palette[4],
                        fg=self.main.color_palette[0],
                        text=i.upper(),
                        anchor=CENTER,
                    )
                )
                text = info[i]
                if len(text) > self.max_length and len(text.split()) > 1:
                    text = text.split()
                    text = text[0]+"\n"+"".join(text[1:])
                self.info_labels.append(
                    Label(
                        self.info_box,
                        bg=self.main.color_palette[2],
                        fg=self.main.color_palette[0],
                        text=text,
                        anchor=CENTER,
                    )
                )
        self.info_box.rowconfigure(0, weight=1)
        self.info_box.columnconfigure(0, weight=1)
        for i, label in enumerate(self.info_labels):
            label.grid(row=i, column=0, sticky=E + W, pady=1, padx=2)

    def set_game_number(self):
        self.game_number["text"] = str(self.main.games.current_game + 1)
        self.set_position_number()
        self.set_info(self.main.games.get_info())

    def set_position_number(self):
        self.position_number["text"] = str(self.main.games.current_move)
        if self.main.games.on_main_move():
            self.position_number["fg"] = self.main.color_palette[4]
            self.position_number["font"] = ("TkDefaultFont", 20)
        else:
            self.position_number["fg"] = self.main.color_palette[2]
            self.position_number["font"] = ("TkDefaultFont", 16)
            
    


class TensorDisplayer(Frame):
    def __init__(self, master, message_width, message_height):
        self.main = master.main
        Frame.__init__(
            self,
            master,
            bg=self.main.color_palette[8],
            width=message_width,
            height=message_height,
        )
        self._create_widgets()
        self.last_selected_button = (0, 0)

    def _create_widgets(self):
        self.tensor_label = Label(self)
        self.exit_button = Button(
            self, command=self.main.stop_display_tensor, text="OK"
        )

        self.tensor_label.place(relx=0.03, rely=0.1, relwidth=0.94, relheight=0.8)
        self.exit_button.place(relx=0.91, rely=0.91, relwidth=0.08, relheight=0.08)

    def set_message(self, message):
        self.tensor_label.config(text=message)
        
class FindOptions(Frame):
    def __init__(self,master,width,height):
        self.main = master.main
        Frame.__init__(
            self,
            master,
            bg=self.main.color_palette[8],
            width=width,
            height=800,
        )
        self._create_widgets(width)
    
    def _create_widgets(self,width):
        
        self.comparison_info = Label(self,text = "Comparsion Type",bg=self.main.color_palette[4]) 
        self.comparison_option = ["Closest vectors","L1 Loos","Cosine distance"]
        self.comparison_selected = StringVar(self)
        self.comparison_selected.set(self.comparison_option[0])
        self.comparison = OptionMenu(self, self.comparison_selected, *self.comparison_option)
        self.number_info = Label(self,text = "amount of games to show",bg=self.main.color_palette[4]) 
        self.number_box = Frame(self, width=width,height = 30,bg=self.main.color_palette[8])
        self.button_box = Frame(self)
        self.submit = Button(self.button_box,text = "Find", command = self.run_coder)
        self.cancel = Button(self.button_box,text = "Cancel", command = self.place_forget)
        self.number = 1

        self.comparison_info.grid(row=0, column=0, sticky=E + W, pady=(10,2),padx=10)
        self.comparison.grid(row=1, column=0, sticky=E + W, pady=2,padx=10)
        self.number_info.grid(row=2, column=0, sticky=E + W, pady=2,padx=10)
        self.number_box.grid(row=3, column=0, sticky=E + W, pady=2,padx=10)
        self.button_box.grid(row=4, column=0, pady=(2,10),padx=10)
        
        self.submit.grid(row=1,column=0, sticky=E + W, padx=2, pady=2)
        self.cancel.grid(row=1,column=1, sticky=E + W, padx=2, pady=2)
        
        size = self.get_size(35)
        image = [
            ImageTk.PhotoImage(
                Image.open(
                    os.path.join(
                        self.main.home_path,
                        "img",
                        str(size) + self.main.arrows_path[i],
                    )
                )
            )
            for i in range(2, 4)
        ]
        
        self.less = Button(
            self.number_box,
            image=image[1],
            activebackground=self.main.color_palette[2],
            command = self.sub_number,
        )
        self.less.image = image[1]

        self.more = Button(
            self.number_box,
            image=image[0],
            activebackground=self.main.color_palette[2],
            command = self.add_number,
        )
        self.more.image = image[0]

        self.number_label = Label(
            self.number_box,
            text="1",
            font=("TkDefaultFont", 16),
        )

        self.less.place(relx=0, rely=0, relwidth=1 / 3, relheight=1)
        self.more.place(relx=2 / 3, rely=0, relwidth=1 / 3, relheight=1)
        self.number_label.place(relx=1/3 + 0.1, rely=0, relwidth=1 / 3 - 0.2, relheight=1)
        
    def get_size(self, num):
        if num <= self.main.images_sizes[0] + self.main.pieces_padding:
            return self.main.images_sizes[0]
        else:
            for i in self.main.images_sizes[1:]:
                if num < i + self.main.pieces_padding:
                    return i
        return self.main.images_sizes[-1]
        
    def add_number(self):
        self.number+=1
        self.number_label["text"] = str(self.number)
    
    def sub_number(self):
        if self.number > 1:
            self.number-=1
        self.number_label["text"] = str(self.number)
        
    def run_coder(self):
        self.main.run_coder(self.number,self.comparison_selected.get())


home_path = os.path.dirname(os.path.abspath(__file__))
pieces_name = "px-Chess_"
pieces_path = {
    "P": "plt",
    "B": "blt",
    "N": "nlt",
    "R": "rlt",
    "Q": "qlt",
    "K": "klt",
    "p": "pdt",
    "b": "bdt",
    "n": "ndt",
    "r": "rdt",
    "q": "qdt",
    "k": "kdt",
}
arrows_path = [
    "px-Arrow_rlt.png",
    "px-Arrow_llt.png",
    "px-Arrow_rdt.png",
    "px-Arrow_ldt.png",
]
images_sizes = [30, 40, 60, 80, 100, 140]
selected_piece = ""
pieces_padding = 10
color_palette = [
    "#2A1F19",
    "#64533C",
    "#F3F3F3",
    "#AF6E63",
    "#C5B0A8",
    "#68493D",
    "#4C413C",
    "#6E635D",
    "#8C8784",
    "#8B644D",
]

window = Tk()
window.iconbitmap('img//icon.ico')
window.title('Chess Autoencoder')
window_height = window.winfo_height()
window_width = window.winfo_width()
window.geometry("900x900")

main_box = Main(
    window,
    color_palette,
    home_path,
    pieces_name,
    pieces_path,
    arrows_path,
    images_sizes,
    selected_piece,
    pieces_padding,
    60,
    90,
    600,
    200,
)


main_box.place(rely=0, relx=0, relwidth=1, relheight=1)

window.mainloop()
