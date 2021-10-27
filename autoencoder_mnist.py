from functions_mnist import prepare_mnist, prepare_autoencoder, load_encoder_decoder
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from datetime import datetime
from pathlib import Path
import sys

resume = False
checkpoint_file_name = ''
start_epoch = 0
epochs = 50

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: <epochs> [checkpoint_file_name start_epoch]')
        exit(1)
    if len(sys.argv) >= 2:
        epochs = sys.argv[1]
    if len(sys.argv) >= 4:
        checkpoint_file_name = sys.argv[2]
        start_epoch = sys.argv[3]
        resume = True


project_name = 'mnist'
base_path = './'
checkpoint_folder = 'checkpoint/'
checkpoint_full_path = base_path + checkpoint_folder + project_name + '/'
Path(checkpoint_full_path).mkdir(parents=True, exist_ok=True)


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

if not resume:
    autoencoder, encoder, decoder = prepare_autoencoder()
else:
    print('Resuming from checkpoint: ' + checkpoint_file_name)
    autoencoder, encoder, decoder = load_encoder_decoder(checkpoint_full_path + checkpoint_file_name)


autoencoder.fit(x_train, [x_train, y_train],
                epochs=epochs,
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

