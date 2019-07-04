from base64 import b64decode
from PIL import Image
from flask import Flask, request, abort
import io

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.form:
        abort(400)

    image_encoded = request.form['image'].split(';')[1].split(',')[1]
    image = Image.open(io.BytesIO(b64decode(image_encoded)))
    pixels = list(image.getdata())
    print(pixels)

    # TODO: predict model
    return '[no model yet]'
