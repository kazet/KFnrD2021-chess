import numpy as np
from IPython.display import HTML, display

import settings
from inference import Inference
from loader import PIECES


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
empty_img = "img/60px_empty.png"
pieces_name = "img/60px-Chess_{}.png"
empty_img_30 = "img/30px_empty.png"
pieces_name_30 = "img/30px-Chess_{}.png"

html = '''<!DOCTYPE html>
<html>
    <head>
        <title></title>
        <meta charset="UTF-8">
        <style>
            .chess-board { border-spacing: 0; border-collapse: collapse; margin-bottom: -1em; }
            .chess-board th { padding: .5em; }
            .chess-board td { border: 1px solid; width: 60px; height: 60px; }
            .chess-board .light { background: whitesmoke; }
            .chess-board .dark { background: sienna }
            .small-table { border-spacing: 0px; border-collapse: none; border-radius: 0px; margin-bottom: -1em; }
            .small-table td { border: 0px none; width: 30px; height: 30px; }
            .small-table tr { border: 0px none; width: 30px; height: 30px; }
            .small-table .light { background: whitesmoke; }
            .small-table .dark { background: sienna }
        </style>
    </head>
    <body>
        <table>
            <tbody>
                <tr>
                    <td style="text-align: center">
                        <h1> Original: </h1>
                        <table class="chess-board">
                            <tbody>
                                %s
                            </tbody>
                        </table>
                        <h2 style="margin-bottom: 0em">castling: %s </h2>
                    </td>
                    <td style="text-align: center">
                        <h1> Reproducted: </h1>
                        <table class="chess-board">
                            <tbody>
                                %s
                            </tbody>
                        </table>
                        <h2 style="margin-bottom: 0em">castling: %s </h2>
                    </td>
                </tr>
            </tbody>
        </table>
    </body>
</html>'''

small_table_temp = '''
    <table class="small-table">
    <tr>
      <td class="{0}"><img src="{1[1]}" style="opacity:{1[0]}; height: 100%"></td>
      <td class="{0}"><img src="{2[1]}" style="opacity:{2[0]}; height: 100%"></td>
    </tr>
    <tr>
      <td class="{0}"><img src="{3[1]}" style="opacity:{3[0]}; height: 100%"></td>
      <td class="{0}"><img src="{4[1]}" style="opacity:{4[0]}; height: 100%"></td>
    </tr>
    </table>'''


def display_fen(fen):
    fen = fen.split(' ')
    board_fen = fen[0]
    player = fen[1]
    table = display_board(board_fen, player == 'w')
    castling = fen[2].replace('K', '♔').replace('Q', '♕').replace('k', '♚').replace('q', '♛')
    return table, castling


def display_board(fen, player):
    table = ''
    fen = fen.split("/")
    if not player:
        fen = fen[::-1]
    for y, i in enumerate(fen):
        table += '<tr>\n'
        x = 0
        for j in i:
            if j.isnumeric():
                img = '<img src="{}">'.format(empty_img)
                for k in range(int(j)):
                    table += '\t<td class="{}">{}</td>\n'.format('dark' if (x + y) % 2 else 'light', img)
                    x += 1
            else:
                img = '<img src="{}">'.format(pieces_name.format(pieces_path[j]))
                table += '\t<td class="{}">{}</td>\n'.format('dark' if (x + y) % 2 else 'light', img)
                x += 1
        table += '</tr>\n'
    return table


def draw_reproduction(fen, model, treshold=0.1, cast_treshold=0.5):
    inf = Inference(settings.DEVICE, model)
    data = inf.predict([fen])
    if data[1].shape == (1,) + settings.BOARD_SHAPE:
        tensor = data[1][0].cpu().detach().numpy()
    else:
        tensor = data[0][0].cpu().detach().numpy()
    board = [[list() for _ in range(8)] for _ in range(8)]
    for brd, pc in zip(tensor, PIECES):
        for x, row in enumerate(zip(*brd)):
            for y, v in enumerate(row):
                if v >= treshold:
                    board[-1-x][y].append((v, pc))
    table = ''
    for y, row in enumerate(board):
        table += '<tr>\n'
        for x, pcs in enumerate(row):
            if len(pcs) > 1:
                pcs.sort(reverse=True)
                tab = []
                for val, name in pcs[:4]:
                    tab.append((val, pieces_name_30.format(pieces_path[name])))
                while len(tab) < 4:
                    tab.append((1, empty_img_30))
                img = small_table_temp.format('dark' if (x + y) % 2 else 'light', *tab)
            elif len(pcs) == 1:
                val, name = pcs[0]
                img = '<img src="{}" style="opacity:{};">'.format(pieces_name.format(pieces_path[name]), val)
            else:
                img = '<img src="{}">'.format(empty_img)
            table += '\t<td class="{}">{}</td>\n'.format('dark' if (x + y) % 2 else 'light', img)
        table += '</tr>\n'
    if fen.split(' ')[1] == 'w':
        cast = ['K', 'Q', 'k', 'q']
    else:
        cast = ['k', 'q', 'K', 'Q']
    cast_str = ''
    for board, letter in zip(tensor[12:], cast):
        if np.mean(board) > cast_treshold:
            cast_str += letter
    if not cast_str:
        cast_str = '-'
    cast_str = ''.join(sorted(cast_str, key=ord))
    cast_str = cast_str.replace('K', '♔').replace('Q', '♕').replace('k', '♚').replace('q', '♛')
    display(HTML(html % (display_fen(fen) + (table, cast_str))))

