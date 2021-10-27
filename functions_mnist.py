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


def prepare_autoencoder():
    # this is our input placeholder
    input = layers.Input(shape=(28, 28, 1))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    x = layers.MaxPool2D((2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D((2, 2), padding='same', name='e')(x)
    e = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    d1 = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same", name='d1')(e)
    d2 = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same", name='d2')(d1)
    d3 = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same", name='d3')(d2)
    c1 = layers.Flatten(name='c1')(e)
    classes = layers.Dense(10, activation='softmax', name='classes')(c1)

    # Autoencoder
    autoencoder = Model(inputs=input, outputs=[d3, classes])
    # plot_model(autoencoder, to_file='autoencoder.png', show_shapes='True')

    encoder = Model(input, e)

    encoded_input = layers.Input(shape=(7, 7, 1))

    # retrieve the last 4 layers of the autoencoder model
    decoder1 = autoencoder.get_layer('d1')
    decoder2 = autoencoder.get_layer('d2')
    decoder3 = autoencoder.get_layer('d3')
    # create the decoder model
    decoder = Model(encoded_input, decoder3(decoder2(decoder1(encoded_input))))

    # First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer:
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()
    encoder.summary()
    decoder.summary()

    return autoencoder, encoder, decoder


def load_encoder_decoder(path):
    autoencoder = load_model(path)

    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('e').output)

    encoded_input = layers.Input(shape=(7, 7, 1))
    x = autoencoder.get_layer('d1')(encoded_input)
    x = autoencoder.get_layer('d2')(x)
    x = autoencoder.get_layer('d3')(x)
    # create the decoder model
    decoder = Model(inputs=encoded_input, outputs=x)

    return autoencoder, encoder, decoder

