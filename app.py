from flask import Flask, render_template, request
import numpy as np
import cv2
from base64 import b64decode
from matplotlib import pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")
@app.route('/frame', methods=['POST'])
def frame():
    json = request.get_json()
    mask = np.array(list(json["mask"].items()))
    mask = mask[:,1].reshape(480,640).astype(np.int)
    mask=np.expand_dims(mask,2)
    header,img_encoded = str(json["img"]).split(",",1)
    img_raw = b64decode(img_encoded)
    img = np.fromstring(img_raw,np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    combined = np.multiply(img,mask)*255
    cv2.imshow("bruh",combined)
    cv2.waitKey(0)
    return "got it"
    
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')