from os import system


system('ffmpeg -i "data\\good posture 2.wmv" -vsync 0 -vf hue=s=0,scale=160:90 "data\\good posture frames\\out%d.png"')
