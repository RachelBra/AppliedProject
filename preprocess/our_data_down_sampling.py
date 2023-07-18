from PIL import Image
import numpy as np
import os

def create_labels(num_classes, num_samples):
    labels = []
    for i in range(num_classes):
        for j in range(num_samples):
            labels.append(i)
    return labels

def down_sampling(load_path):
    image = Image.open(load_path)
    resized_image = image.resize((32, 32), Image.LANCZOS)
    return resized_image

def create_images(folder_load_path):
    images = []
    for file_name in os.listdir(folder_load_path):
        load_path = os.path.join(folder_load_path, file_name)
        # save_path = os.path.join(folder_save_path, "our_data")
        resized_image = down_sampling(load_path)
        images.append(np.array(resized_image))
    return images


folder_load_path = os.getcwd()+"\\data\\our data\\our_data_umge from _google" # change to your path
folder_save_path = os.getcwd()+"\\data\\our data"

num_classes = 10
num_samples = 5
labels = create_labels(num_classes, num_samples)
images = create_images(folder_load_path)

print("labels", len(labels))
print("images", len(images))

labels_data = np.array(labels)
images_data = np.vstack(images)

our_data_dict = {'images': images_data, 'labels': labels_data}
np.savez(folder_save_path, **our_data_dict)
