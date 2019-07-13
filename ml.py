from tensorflow.python.keras import activations
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
import dataset
import numpy as np


def get_tensor_dir(opt: str, epochs: int,
                   lr: float, decay:float,
                   l_inputs: int, test_mode=False):

    dir_name = "./logs/d" + str(l_inputs) + "_lr" + str(lr) + "_dc" + str(decay) + "_e" + str(epochs)

    if test_mode:
        dir_name += "_test"
    return dir_name


def get_file_path(optimizer: str, l_inputs: int, lr: float):
    return "./models/" + optimizer + "_d" + str(l_inputs) + "_lr" + str(lr) + "_e{epoch:03d}_acc{acc:.2f}_vacc{val_acc:.2f}.h5"


if __name__ == "__main__":
    # Dataset build
    x_man, y_man = dataset.load_data("dataset/man", "png", 0)
    x_woman, y_woman = dataset.load_data("dataset/woman", "png", 1)

    x_data = np.concatenate((x_man, x_woman)) / 255.0
    y_data = np.concatenate((y_man, y_woman))

    x_data, y_data = dataset.randomize(x_data, y_data)

    # Model
    model = models.Sequential()

    # Layers
    layer_inputs = 32
    model.add(layers.Dense(layer_inputs,
                           input_shape=(3072, ),
                           activation=activations.relu))
    model.add(layers.Dense(1,
                           activation=activations.sigmoid))

    # Compile
    learning_rate = .15
    decay = .005
    opt = "sgd"
    optimizer = optimizers.SGD(lr=learning_rate, decay=decay)

    model.compile(loss=losses.binary_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Callbacks
    epochs = 300
    tensor_cb = callbacks.TensorBoard(log_dir=get_tensor_dir(opt=opt,
                                                             epochs=epochs,
                                                             lr=learning_rate,
                                                             decay=decay,
                                                             l_inputs=layer_inputs))

    #early_cb = callbacks.EarlyStopping(monitor='val_acc', patience=5, min_delta=0.01)

    checkpoint_cp = callbacks.ModelCheckpoint(filepath=get_file_path(optimizer=opt, l_inputs=layer_inputs, lr=learning_rate),
                                              monitor="val_acc",
                                              save_best_only=True,
                                              period=3)

    model.fit(x_data, y_data,
              epochs=epochs,
              batch_size=64,
              validation_split=0.2,
              callbacks=[tensor_cb])
