import cv2
import requests
import numpy as np
from io import BytesIO
from flask import Flask, render_template, request, send_file

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    image_url = request.form['image_url']
    response = requests.get(image_url)
    image_bytes = BytesIO(response.content)
    img = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)

    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(gr)
    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    invertBlur = cv2.bitwise_not(blur)
    sketch = cv2.divide(gr, invertBlur, scale=256.0)

    cv2.imwrite('../sketchify/sketch.jpg', sketch)

    return send_file('../sketchify/sketch.jpg', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
