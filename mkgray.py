from os import system


system('ffmpeg -i "data\\bad posture.wmv" -vsync 0 -vf hue=s=0,scale=160:90 "data\\test\\bad\\out%d.png"')
