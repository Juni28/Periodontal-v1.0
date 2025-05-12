# evaluate.py

import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES

# Métricas
def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = np.sum(intersection) / (np.sum(union) + 1e-7)
    return iou

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    dice = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)
    return dice

# Cargar modelo entrenado
model = load_model("unet_periodontal.h5", compile=False)

# Directorios
image_dir = "Dataset/Validation/Images"
mask_dir = "Dataset/Validation/Masks"

image_paths = sorted([os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith(".jpg")])
mask_paths = sorted([os.path.join(mask_dir, x) for x in os.listdir(mask_dir) if x.endswith(".png")])

iou_scores = []
dice_scores = []

print("Evaluando...")

for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
    # Cargar imagen
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Cargar máscara real
    mask = load_img(mask_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale")
    mask = img_to_array(mask).astype("int32")
    mask = np.squeeze(mask)
    mask = tf.keras.utils.to_categorical(mask, num_classes=NUM_CLASSES)

    # Predicción
    pred = model.predict(img)[0]
    pred_mask = np.argmax(pred, axis=-1)
    true_mask = np.argmax(mask, axis=-1)

    # Calcular métricas
    iou_scores.append(iou_score(true_mask, pred_mask))
    dice_scores.append(dice_score(true_mask, pred_mask))

# Resultados
print("\n--- Resultados ---")
print(f"IoU promedio: {np.mean(iou_scores):.4f}")
print(f"Dice promedio: {np.mean(dice_scores):.4f}")
