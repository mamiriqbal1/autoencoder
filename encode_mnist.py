from functions_mnist import load_encoder_decoder, prepare_mnist
import numpy as np
from pathlib import Path
import sys


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: <checkpoint_file_name>')
        exit(1)
    if len(sys.argv) >= 2:
        checkpoint_file_name = sys.argv[1]


(x_train, y_train), (x_test, y_test) = prepare_mnist()

project_name = 'mnist'
base_path = './'
checkpoint_folder = 'checkpoint/'
checkpoint_full_path = base_path + checkpoint_folder + project_name + '/'
data_path = 'data/'
data_path_full = base_path + data_path + project_name + '/'
Path(data_path_full).mkdir(parents=True, exist_ok=True)


autoencoder, encoder, decoder = load_encoder_decoder(checkpoint_full_path + checkpoint_file_name)
x_train_encoded = encoder.predict(x_train)
x_test_encoded = encoder.predict(x_test)

# normalize between 0 and 1
x_train_encoded = (x_train_encoded - x_train_encoded.min()) / (x_train_encoded.max() - x_train_encoded.min())
x_test_encoded = (x_test_encoded - x_test_encoded.min()) / (x_test_encoded.max() - x_test_encoded.min())

x_train_encoded = np.reshape(x_train_encoded, (x_train_encoded.shape[0], x_train_encoded.shape[1]*x_train_encoded.shape[2]))
x_test_encoded = np.reshape(x_test_encoded, (x_test_encoded.shape[0], x_test_encoded.shape[1]*x_test_encoded.shape[2]))

np.savetxt(data_path_full + 'train_encoded_mnist.txt', x_train_encoded)
np.savetxt(data_path_full + 'test_encoded_mnist.txt', x_test_encoded)

np.savetxt(data_path_full + 'train_label_mnist.txt', y_train, fmt='%d')
np.savetxt(data_path_full + 'test_label_mnist.txt', y_test, fmt='%d')
