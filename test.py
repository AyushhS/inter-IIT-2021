import sys
import os

Videopath = str(sys.argv[1])

outputfolder = []
if len(sys.argv) == 1:
    os.mkdir('output')
    outputfolder = 'output'
else:
    if os.path.isdir(sys.argv[2]) == True:
        outputfolder = sys.argv[2]
    else:
        outputfolder = os.mkdir(sys.argv[2])

if os.path.isfile(Videopath) == True:
    videos = [Videopath]
else:
    videos = os.listdir(Videopath)