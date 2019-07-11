from tensorflow.python.keras import activations
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
import dataset
import numpy as np


if __name__ == "__main__":
    x_man, y_man = dataset.load_data("dataset/man", "png", 0)
    x_woman, y_woman = dataset.load_data("dataset/woman", "png", 1)

    x_data = np.concatenate((x_man, x_woman)) / 255.0
    y_data = np.concatenate((y_man, y_woman))

    x_data, y_data = dataset.randomize(x_data, y_data)

    # Model
    model = models.Sequential()

    # Layers
    model.add(layers.Dense(1,
                           input_shape=(3072, ),
                           activation=activations.sigmoid))

    model.compile(loss=losses.binary_crossentropy,
                  optimizer=optimizers.sgd(),
                  metrics=['accuracy'])

    tensor_cb = callbacks.TensorBoard(log_dir="./logs/1")

    model.fit(x_data, y_data,
              epochs=100,
              batch_size=64,
              validation_split=0.2,
              verbose=1,
              callbacks=[tensor_cb])

    model.save('./models/sgd_C1_E100_B64.h5')
