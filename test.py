from GeneratorPgn import IteratorPgn, Preprocessor, ArrayToFen
import chess
import numpy as np
"""
tests whether the downloaded fen record after processing into a list and then restored again has not been deformed, and whether the given fen record may mean a real position
"""
number_of_tests=1000
test_number=0
batch_in_test=1
loader=Preprocessor(IteratorPgn(batch_in_test, [["lichessPrepared.pgn"],["lichess.pgn.bz2"]], 100, 10))
result=0
for array,fens in loader :
    try:
        data=ArrayToFen(array)
        for x in range(len(data)):
            test_board=chess.Board().set_fen(data[x])
            if(data[x]==fens[x]):
                result+=1
            else:
                print('')
                print('incorrect data', data[x],fens[x],sep='\n')
        test_number+=1
    except Exception as e:
        print('')
        print(e,"","An error occurred with this data",array,fens,sep='\n')
    print('\r',"progres - ",str(test_number),"/",str(number_of_tests),end='')
    if(number_of_tests==test_number):
        break
print('')
print("Percentage of Correct Transcription - "+str(result/number_of_tests/batch_in_test*100)+"%")