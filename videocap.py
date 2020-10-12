import cv2
import numpy as np
import time

predicting=True
writing=False
if predicting:
    from model_example import make_model
    model = make_model(input_shape=(80,60) + (1,), num_classes=2)
    model.load_weights("save_at_10.h5")

vc=cv2.VideoCapture(0)
cv2.namedWindow("preview")
width=80
height=60
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
def capture(directory):
    while True:
        rval, frame = vc.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("preview", frame)
        frame= cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
        if writing:
            path='data\\{directory}\\img{time}.jpg'.format(time=round(time.time()*1000),directory=directory)
            cv2.imwrite(path,frame)
        if predicting:
            frame=np.expand_dims(frame,axis=0)
            print(model.predict(frame))
        key = cv2.waitKey(500)
        if key == 32 or key == 27 or key == 13: # exit on space or esc or enter
            break

while(rval):
    key=cv2.waitKey(0)
    if key == 13:#enter
        capture('bad')
    if key == 32:#space
        capture('good')
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")
vc.release()
