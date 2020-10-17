from flask import Flask, render_template, request
import numpy as np
import cv2
from base64 import b64decode

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")
@app.route('/frame', methods=['POST'])
def frame():
    data_uri = request.get_data()
    header,encoded = str(data_uri).split(",",1)
    data=b64decode(encoded)
    nparr = np.fromstring(data,np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
    print(img_np.shape)
    return "got it"
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')