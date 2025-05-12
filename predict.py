import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES

# --- Cargar el modelo entrenado ---
model = load_model("unet_periodontal.h5")
print("Modelo cargado.")

# --- Cargar una imagen desde disco ---
def load_image(image_path):
    image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    image_array = img_to_array(image) / 255.0  # Normalizar
    return image_array

# --- Superponer la máscara sobre la imagen original ---
def show_overlay(original, mask_pred, alpha=0.4):
    image = (original * 255).astype(np.uint8)

    # Convertir máscara a colores
    mask_pred = np.argmax(mask_pred, axis=-1).squeeze()
    color_mask = np.zeros_like(image)

    # Colores para cada clase (puedes personalizar)
    colors = {
        1: [255, 0, 0],   # Rojo
        2: [0, 255, 0],   # Verde
        3: [0, 0, 255],   # Azul
        4: [255, 255, 0], # Amarillo
        5: [255, 0, 255], # Magenta
        6: [0, 255, 255], # Cyan
    }

    for class_id, color in colors.items():
        color_mask[mask_pred == class_id] = color

    blended = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(blended)
    plt.title("Segmentación superpuesta")
    plt.axis('off')
    plt.show()

# --- Ruta de la imagen a probar ---
image_path = "Dataset/Test/Images/00929.jpg"  # Cambia este nombre por otra imagen si deseas

# --- Procesar y predecir ---
image_array = load_image(image_path)
input_image = np.expand_dims(image_array, axis=0)
prediction = model.predict(input_image)

# --- Mostrar el resultado ---
show_overlay(image_array, prediction)
