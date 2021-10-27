from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist
import numpy as np


def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array


def prepare_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)
    return (x_train, y_train), (x_test, y_test)


def load_encoder_decoder(path):
    autoencoder = load_model(path)

    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('e').output)

    encoded_input = layers.Input(shape=(7, 7, 1))
    x = autoencoder.get_layer('d1')(encoded_input)
    x = autoencoder.get_layer('d2')(x)
    x = autoencoder.get_layer('d3')(x)
    # create the decoder model
    decoder = Model(inputs=encoded_input, outputs=x)

    return encoder, decoder

