from tkinter import *
from PIL import ImageTk, Image
from functools import partial
import os
import chess
from tkinter import filedialog

from inference import Inference


class Main(Frame):
    def __init__(
        self,
        master,
        header_height,
        option_width,
        color_palette,
        home_path,
        pieces_name,
        pieces_path,
        arrows_path,
        images_sizes,
        selected_piece,
        pieces_padding,
    ):
        Frame.__init__(self, master, bg=color_palette[0])
        master.rowconfigure(0, weight=1)
        master.columnconfigure(0, weight=1)
        self.main = self
        self.home_path = home_path
        self.pieces_name = pieces_name
        self.pieces_path = pieces_path
        self.arrows_path = arrows_path
        self.images_sizes = images_sizes
        self.selected_piece = selected_piece
        self.pieces_padding = pieces_padding
        self.color_palette = color_palette
        self.fen_placement = "8/8/8/8/8/8/8/8"
        self.fen_player = "w"
        self.fen_castling = "-"
        self.header_height = header_height
        self.option_width = option_width
        self._create_widgets()
        self.bind("<Configure>", self._resize)
        self.winfo_toplevel().minsize(400, 400)
        self.display_fen()
        self.coder = None
        self.coder_launcher = None

    def _create_widgets(self):
        self.board_box = BoardBox(self)
        self.option_box = Options(self, self.option_width)
        self.header = Header(self, header_height=self.header_height)

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

    def set_fen(self, fen):
        try:
            a = chess.Board(fen)
            del a
            split_fen = fen.split()
            self.fen_placement = split_fen[0]
            self.fen_player = split_fen[1]
            self.fen_castling = split_fen[2]
            self.option_box.set_option(self.fen_player, self.fen_castling)
            self.board_box.board.set_board(self.fen_placement)
        except ValueError:
            self.header.display_fen("Incorrect fen", "", "")

    def set_coder(self, filename):
        try:
            self.coder = torch.load(filename)
            self.coder.eval()
            device = torch.device(
                "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
            )
            self.coder_launcher = Inference(
                device,
                self.coder,
            )
            return True
        except:
            return False

    def run_coder(self):
        print("Results : \n", self.coder_launcher.predict(fens))


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
            command=self.main.run_coder,
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
        self.main.set_fen(self.fen_entry.get())

    def get_coder(self):
        filename = filedialog.askopenfilename(
            filetypes=[("pickle format", "*.pt"), ("pytorch compiled", "*.pth")]
        )
        if self.main.set_coder(filename):
            self.coder_label["text"] = "Coder set"
        else:
            self.coder_label["text"] = "Coder couldn't be set"


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
window_height = window.winfo_height()
window_width = window.winfo_width()
window.geometry("900x900")

main_box = Main(
    window,
    60,
    100,
    color_palette,
    home_path,
    pieces_name,
    pieces_path,
    arrows_path,
    images_sizes,
    selected_piece,
    pieces_padding,
)


main_box.place(rely=0, relx=0, relwidth=1, relheight=1)

window.mainloop()
