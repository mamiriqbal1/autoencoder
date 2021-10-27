import numpy as np
import sys


in_base_path = 'data/mnist/'
out_base_path = 'data/mnist/'

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage: <data_file_name, label_file_name, out_file_name>')
        exit(1)
    if len(sys.argv) >= 4:
        data_file_name = sys.argv[1]
        label_file_name = sys.argv[2]
        out_file_name = sys.argv[3]


data_path = in_base_path + data_file_name
label_path = in_base_path + label_file_name
out_path = out_base_path + out_file_name


def prepare_mnist_encoded(data_path, label_path, out_path):
    data = np.loadtxt(data_path)
    label = np.loadtxt(label_path)
    label = np.reshape(label, (label.shape[0], 1))
    encoded_mnist = np.concatenate((data, label), axis=1)
    np.savetxt(out_path, encoded_mnist)
    print('done')


prepare_mnist_encoded(data_path, label_path, out_path)

