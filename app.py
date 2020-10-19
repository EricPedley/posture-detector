from flask import Flask, render_template, request
import numpy as np
import cv2
from base64 import b64decode
import random
width=80
height=int(width*3/4)
predicting=True
writing=True
if writing:
    import time
if predicting:
    from model import make_model
    model = make_model(input_shape=(width,height) + (1,), num_classes=2)
    model.load_weights("model_weights.h5")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")
@app.route('/frame', methods=['POST'])
def frame():
    json = request.get_json()
    mask = np.array(list(json["mask"].items()))
    mask = mask[:,1].reshape(480,640).astype(np.uint8)
    header,img_encoded = str(json["img"]).split(",",1)
    img_raw = b64decode(img_encoded)
    img = np.fromstring(img_raw,np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    combined = np.multiply(img,mask)
    frame = cv2.resize(combined,(width,height),interpolation=cv2.INTER_AREA)
    if writing:
            if json["mode"] != "predict":
                directory=json["mode"]
                path='data\\{directory}\\img{time}.jpg'.format(time=round(time.time()*1000),directory=directory)
                cv2.imwrite(path,frame)
    if predicting:
            frame=np.expand_dims(frame,axis=2)
            frame=np.expand_dims(frame,axis=0)
            frame=np.swapaxes(frame,1,2)
            final_prediction=model.predict(frame)[0][0]
            return str(round(final_prediction,3))
    return str(round(random.random(),3))
    
if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')