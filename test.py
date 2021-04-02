import GeneratorPgn

# Test
obj = GeneratorPgn.IteratorPgn(200, [["lichessPrepared.pgn"],["lichess.pgn.bz2"]], 600, 100)
for i in obj:
    print(i)
