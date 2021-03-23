'''
    path - path to .pgn file
    maxGame - maximum number of games to parse
'''
def pgnParse(path,maxGame=None):
    result=[{}]
    with open(path,'r') as f:
        for line in f:
            #tags begin with "["
            #Movetext begin with "1."
            if(line[0]=="["):
                j=line.split(" ")
                result[-1][j[0][1:]]=j[1][1:-2]
            elif(line[0]=="1"):
                game=[]
                for j in line.split(". ")[1:]:
                    #there are several kinds of Movetext notation, so also several kinds of split
                    if(" {" in j):
                        moves=j.split(" {")
                        for i in moves[:-1]:
                            game.append(i.split(" ")[-1].replace("!","").replace("!?","").replace("?",""))
                    else:
                        moves=j.split(" ")
                        game.append(moves[0])
                        if(("\n" in moves[1])==False and (moves[1].isdigit())==False):
                            game.append(moves[1].replace("!","").replace("!?","").replace("?",""))
                result[-1]["Movetext"]=game
                if(maxGame!=None and maxGame<=len(result)):
                    break
                result.append({})
            # Stopping the function when we have received a given number of games
            
    return result

'''
>>> pgnParse("lichess2.pgn",2)
[{'Event': 'Rat', 'Site': 'https://lichess.org/tGpzk7yJ"', 'White': 'calvinmaster"', 'Black': 'dislikechess"', 'Result': '1-0"', 'UTCDate': '2017.03.31"', 'UTCTime': '22:00:01"', 'WhiteElo': '2186"', 'BlackElo': '1907"', 'WhiteRatingDiff': '+4"', 'BlackRatingDiff': '-4"', 'ECO': 'C34"', 'Opening': 'King', 'TimeControl': '180+0"', 'Termination': 'Normal"', 'Movetext': ['e4', 'e5', 'f4', 'exf4', 'Nf3', 'Nf6', 'e5', 'Nh5', 'Bc4', 'g5', 'h4', 'Ng3', 'Nxg5', 'Nxh1', 'Bxf7+', 'Ke7', 'Nc3', 'c6', 'd4', 'h6', 'Qh5', 'Bg7', 'Nge4', 'Qf8', 'Nd6', 'Na6', 'Bxf4', 'Nb4', 'Kd2', 'Nf2', 'Rf1', 'Rh7', 'Rxf2', 'Bh8', 'Bg5+', 'hxg5', 'Qxg5+']}, {'Event': 'Rat', 'Site': 'https://lichess.org/LzvBtZ93"', 'White': 'Gregster101"', 'Black': 'flavietta"', 'Result': '1-0"', 'UTCDate': '2017.03.31"', 'UTCTime': '22:00:00"', 'WhiteElo': '1385"', 'BlackElo': '1339"', 'WhiteRatingDiff': '+10"', 'BlackRatingDiff': '-9"', 'ECO': 'C34"', 'Opening': 'King', 'TimeControl': '120+1"', 'Termination': 'Ti', 'Movetext': ['e4', 'e5', 'f4', 'exf4', 'Nf3', 'Bc5', 'd4', 'Bb6', 'Bxf4', 'Nf6', 'e5', 'Qe7', 'Bc4', 'O-O', 'O-O', 'Ng4', 'c3', 'd6', 'Nbd2', 'Nc6', 'Re1', 'Bf5', 'exd6', 'cxd6', 'Rxe7', 'Nxe7', 'Nh4', 'Bd7', 'Bxd6', 'Rfe8', 'Qf3', 'Nf6', 'Re1', 'Ned5', 'Rxe8+', 'Rxe8', 'Bxd5', 'Re1+', 'Kf2', 'Ra1', 'Ne4', 'Ng4+', 'Kg3', 'Nf6', 'Nxf6+', 'gxf6', 'Qxf6']}]
'''