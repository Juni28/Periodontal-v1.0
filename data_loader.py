# data_loader.py

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES

def load_dataset(image_dir, mask_dir):
    image_files = sorted(os.listdir(image_dir))
    images = []
    masks = []
    
    for filename in image_files:
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename.replace('.jpg', '.png'))

        if not os.path.exists(mask_path):
            continue

        img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        mask = load_img(mask_path, color_mode='grayscale', target_size=(IMG_HEIGHT, IMG_WIDTH))

        images.append(img_to_array(img) / 255.0)
        masks.append(img_to_array(mask).squeeze().astype(np.uint8))

    X = np.array(images)
    y = np.array(masks)
    y = np.eye(NUM_CLASSES)[y]  # one-hot encoding

    return train_test_split(X, y, test_size=0.2, random_state=42)
