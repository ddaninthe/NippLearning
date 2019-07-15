from base64 import b64decode
from PIL import Image
from flask import Flask, request, abort
from tensorflow.python.keras import models
import numpy as np
from math import isclose
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

    # Remove alpha
    image = image.convert('RGB')

    image = np.reshape(image, (1, 3072))

    model_name = "sgd_d64.32.1_lr0.2_e120_acc0.84_vacc0.65.h5"
    model = models.load_model("./models/" + model_name)

    value = model.predict(image)

    if isclose(value, 0.0, rel_tol=1e-6):
        prediction = "Un Homme"
    else :
        prediction = "Une Femme"

    return prediction
