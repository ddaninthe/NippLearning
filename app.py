from base64 import b64decode
from PIL import Image
from flask import Flask, request, abort
from tensorflow.python.keras import models
import numpy as np
from math import isclose
import io
import json

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.form:
        abort(400)

    results = []
    request_images = json.loads(request.form['images'])
    for form_image in request_images:
        img_name = form_image['name']

        image_encoded = form_image['src'].split(';')[1].split(',')[1]
        image = Image.open(io.BytesIO(b64decode(image_encoded)))

        # Remove alpha
        image = image.convert('RGB')

        image = np.reshape(image, (1, 3072))

        model_name = "sgd_d64.32.1_lr0.2_e120_acc0.84_vacc0.65.h5"
        model = models.load_model("./models/" + model_name)

        value = model.predict(image)

        if isclose(value, 0.0, rel_tol=1e-6):
            results.append([img_name, "Homme"])
        else:
            results.append([img_name, "Femme"])

    return json.dumps(results)
