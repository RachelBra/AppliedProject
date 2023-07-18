import pickle
import numpy as np
import os
from PIL import Image


def save_cifar10_as_numpy_dict(data_dir, output_file):
    train_data, train_labels = [], []
    for i in range(1, 6):
        file_path = os.path.join(data_dir, f'data_batch_{i}')
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            train_labels.extend(data[b'labels'])
            for image_data in data[b'data']:
                image = np.reshape(image_data, (32, 32, 3))
                resized_image = Image.fromarray(image).resize((32, 32), Image.LANCZOS)
                rgb_image = resized_image.convert("RGB")
                train_data.append(np.array(rgb_image))

    print("images", len(train_data))
    print("labels", len(train_labels))

    train_data = np.array(train_data)
    print("images after", train_data.shape)

    print("images after", len(train_data[0]))
    train_labels = np.array(train_labels)
    cifar_dict = {'images': train_data, 'labels': train_labels}

    np.savez(output_file, **cifar_dict)


path = os.getcwd()+r'\\data\\cifr10\\cifar-10-batches-py'  #your path
output_file = os.getcwd() + r'\\data\\cifr10'
save_cifar10_as_numpy_dict(path, output_file)













