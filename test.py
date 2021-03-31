import Generator_pgn

# Test
obj = Generator_pgn.Iterator_pgn(200, [["lichessPrepared.pgn"],["lichess.pgn.bz2"]], 600, 100)
for i in obj:
    print(i)
