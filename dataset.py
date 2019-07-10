from PIL import Image
import glob
import numpy


def load_data(directory, extension_type, value):
    x_train = numpy.array([])
    i = 0
    for filename in glob.glob(directory + "/*." + extension_type):
        i += 1
        img = Image.open(filename)
        x_train = numpy.append(x_train, img)

    x_train = numpy.reshape(x_train, (i, 32 * 32 * 3))
    y_train = numpy.array([])
    for j in range(0, i):
        y_train = numpy.append(y_train, value)

    return x_train, y_train


def randomize(x_data, y_data):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(x_data)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(y_data)

    return x_data, y_data
