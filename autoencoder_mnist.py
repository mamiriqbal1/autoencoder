from functions_mnist import prepare_mnist
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from datetime import datetime
from pathlib import Path

project_name = 'mnist'
base_path = './'
checkpoint_folder = 'checkpoint/'
checkpoint_full_path = base_path + checkpoint_folder + project_name + '/'
Path(checkpoint_full_path).mkdir(parents=True, exist_ok=True)
start_epoch = 0


def prepare_autoencoder():
    # this is our input placeholder
    input = layers.Input(shape=(28, 28, 1))

    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input)
    x = layers.MaxPool2D((2, 2), padding='same')(x)
    x = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    e = layers.MaxPool2D((2, 2), padding='same', name='e')(x)

    # Decoder
    d1 = layers.Conv2DTranspose(1, (3, 3), strides=2, activation="relu", padding="same", name='d1')(e)
    d2 = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same", name='d2')(d1)
    d3 = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same", name='d3')(d2)
    c1 = layers.Flatten(name='c1')(d2)
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


(x_train, y_train), (x_test, y_test) = prepare_mnist()

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Now let's train our autoencoder
now = datetime.now()
log_str = now.strftime('%Y%m%d%H%M%S') + '-' + str(start_epoch)
checkpoint = ModelCheckpoint(filepath=checkpoint_full_path + log_str + '-checkpoint-{epoch:03d}-{loss:.6f}.h5',
                             monitor='loss', verbose=0, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

autoencoder, encoder, decoder = prepare_autoencoder()
autoencoder.fit(x_train, [x_train, y_train],
                epochs=50,
                verbose=2,
                batch_size=50,
                callbacks=callbacks_list,
                shuffle=True)


# After 50 epochs, the autoencoder seems to reach a stable train/test loss value of about 0.11.
# We can try to visualize the reconstructed inputs and the encoded representations. We will use Matplotlib.
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_train)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig('digits.png')

