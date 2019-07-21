from tensorflow.python.keras.activations import *
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
import dataset
import numpy as np


def get_tensor_dir(opt: str, epochs: int,
                   l_inputs=[], lr: float = 0,
                   decay: float = 0, mom: float = 0,
                   test_mode=False):

    dir_name = "./logs/" + opt + "_d"
    for li in l_inputs:
        dir_name += str(li) + "."

    dir_name = dir_name[:-1]

    if lr != 0:
        dir_name += "_lr" + str(lr)

    if decay != 0:
        dir_name += "_dc" + str(decay)
    dir_name += "_e" + str(epochs)

    if mom != 0:
        dir_name += "_m" + str(mom)

    if test_mode:
        dir_name += "_test"
    return dir_name


def get_file_path(optimizer: str, l_inputs=[], lr: float = 0, mom: float = 0):
    file_path = "./models/" + optimizer + "_d"\

    for li in l_inputs:
        file_path += str(li) + "."

    file_path += "1"
    if lr != 0:
        file_path += "_lr" + str(lr)

    if mom != 0:
        file_path += "_m" + str(mom)

    file_path += "_e{epoch:03d}_acc{acc:.2f}_vacc{val_acc:.2f}.h5"
    return file_path


if __name__ == "__main__":
    # Dataset build
    x_man, y_man = dataset.load_data("dataset/man", "png", 0)
    x_woman, y_woman = dataset.load_data("dataset/woman", "png", 1)

    x_data = np.concatenate((x_man, x_woman)) / 255.0
    y_data = np.concatenate((y_man, y_woman))

    x_data, y_data = dataset.randomize(x_data, y_data)

    # Model
    model = models.Sequential()

    epochs = 150

    # Layers
    layer_inputs = [512, 64, 32]
    for i in range(len(layer_inputs)):
        if i == 0:
            model.add(Dense(layer_inputs[0],
                            input_shape=(3072, ),
                            activation=relu))
        else:
            model.add(Dense(layer_inputs[i],
                            activation=relu))

    '''ConvNet
    model.add(Reshape((32, 32, 3), input_shape=(3072,)))
    model.add(Conv2D(16, 3, padding='same', activation=relu))
    model.add(AveragePooling2D())
    model.add(Conv2D(32, 3, padding='same', activation=relu))
    model.add(AveragePooling2D())
    model.add(Conv2D(64, 3, padding='same', activation=relu))
    model.add(AveragePooling2D())
    model.add(Flatten())
    '''

    model.add(Dense(1, activation=sigmoid))

    # Compile
    opt = "sgd"
    momentum = 0.5
    optimizer = optimizers.SGD(momentum=momentum)

    model.compile(loss=losses.binary_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Callbacks
    log_dir = get_tensor_dir(opt=opt,
                             epochs=epochs,
                             l_inputs=layer_inputs,
                             mom=momentum,
                             test_mode=False)
    tensor_cb = callbacks.TensorBoard(log_dir=log_dir)

    # Stops the training if no improvement
    early_cb = callbacks.EarlyStopping(monitor='val_acc',
                                       patience=20,
                                       min_delta=0.001)

    # Saves model while training if [monitor] param is better than the [period] previous epochs
    file_path = get_file_path(optimizer=opt,
                              l_inputs=layer_inputs,
                              mom=momentum)
    checkpoint_cp = callbacks.ModelCheckpoint(filepath=file_path,
                                              monitor="val_acc",
                                              save_best_only=True,
                                              period=1)

    model.fit(x_data, y_data,
              epochs=epochs,
              batch_size=64,
              validation_split=0.2,
              callbacks=[tensor_cb, checkpoint_cp])

    model.save("./models/sgd_d512.64.32_e150_m.5.h5", overwrite=True)
