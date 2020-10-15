import cv2
import numpy as np
import time
from matplotlib import pyplot

predicting=True
writing=False
depth=True

net=sess=input_node=predictDepth=(None,None,None,None)

if depth:
    from depth_predict import predict
    predictDepth=predict.predict
    net,sess,input_node=predict.init_net("depth_predict\\NYU_FCRN-checkpoint\\NYU_FCRN.ckpt")

if predicting:
    from model_example import make_model
    model = make_model(input_shape=(80,60) + (1,), num_classes=2)
    model.load_weights("save_at_10.h5")
    print(model.input_shape)

vc=cv2.VideoCapture(0)
cv2.namedWindow("raw")
backsub = cv2.createBackgroundSubtractorKNN()#createBackgroundSubtractorMOG2()

width=80
height=60
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
def capture(directory):
    while True:
        rval, frame = vc.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #fgmask = backsub.apply(frame)
        cv2.imshow("raw", frame)
        if depth:
            frame = cv2.resize(frame,(304,228),interpolation=cv2.INTER_AREA)
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame=np.expand_dims(frame,0)
            prediction = predictDepth(net,sess,input_node,frame)[0,:,:,0]#128x160
            #print(prediction[0][0])
            #prediction=(prediction/256).astype(np.uint8)
            prediction=cv2.resize(prediction,(304,228),interpolation=cv2.INTER_AREA)
            prediction = (1/(prediction/np.amin(prediction)))
            mask=prediction-0.2#closest values are 1, so anything above 0.5+x is included
            mask=np.round(mask)
            #frame = np.multiply(gray,prediction)
            masked = np.multiply(gray,mask)
            #pyplot.imshow(masked)
            #pyplot.show()
            frame = masked#np.concatenate((np.expand_dims(gray,2), np.expand_dims(masked,2),np.expand_dims(prediction*255,2)),axis=2)
            #cv2.imshow("highlighted depth",prediction/255)
            cv2.imshow("combined",frame/255)
            #cv2.imshow("depthmap",prediction)    
        frame= cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
        if writing:
            path='data\\{directory}\\img{time}.jpg'.format(time=round(time.time()*1000),directory=directory)
            cv2.imwrite(path,frame)
        if predicting:
            frame=np.expand_dims(frame,axis=2)
            frame=np.expand_dims(frame,axis=0)
            frame=np.swapaxes(frame,1,2)
            final_prediction=model.predict(frame)[0][0]
            print(final_prediction)
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
