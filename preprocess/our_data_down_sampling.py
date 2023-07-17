from PIL import Image
import numpy as np

import os

def down_sampling(load_path, save_path):
    image = Image.open(load_path)
    resized_image = image.resize((32, 32), Image.LANCZOS)

    umpydata = np.asarray(resized_image)
    np.save(save_path, umpydata)




folder_load_path = r'C:\bootcamp\project\our_data_images'  # change to your path
folder_save_path = r"C:\bootCamp\project\our_data"
for file_name in os.listdir(folder_load_path):
    load_path = os.path.join(folder_load_path, file_name)
    save_path = os.path.join(folder_save_path, "our_data")
    down_sampling(load_path, save_path)


