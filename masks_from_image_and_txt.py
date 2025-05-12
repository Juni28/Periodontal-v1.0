import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_all_yolo_to_masks(base_dir, subsets=['Training', 'Test', 'Validation'], num_classes=5):
    for subset in subsets:
        print(f"\nProcesando {subset}...")
        image_dir = os.path.join(base_dir, subset, "Images")
        label_dir = os.path.join(base_dir, subset, "Labels")
        mask_dir = os.path.join(base_dir, subset, "Masks")

        os.makedirs(mask_dir, exist_ok=True)

        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

        for img_name in tqdm(image_files):
            img_path = os.path.join(image_dir, img_name)
            label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
            mask_path = os.path.join(mask_dir, img_name.replace('.jpg', '.png'))

            # Cargar imagen para dimensiones
            img = Image.open(img_path)
            w, h = img.size

            # Crear máscara vacía
            mask = np.zeros((h, w), dtype=np.uint8)

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue  # Saltar líneas mal formateadas
                        class_id, x_center, y_center, box_w, box_h = map(float, parts)
                        class_id = int(class_id)

                        x1 = int((x_center - box_w / 2) * w)
                        y1 = int((y_center - box_h / 2) * h)
                        x2 = int((x_center + box_w / 2) * w)
                        y2 = int((y_center + box_h / 2) * h)

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w - 1, x2), min(h - 1, y2)

                        # Rellenar la máscara con la clase
                        cv2.rectangle(mask, (x1, y1), (x2, y2), class_id, -1)

            # Guardar la máscara
            Image.fromarray(mask).save(mask_path)

# 🔧 Ejecutar el script
dataset_path = "Dataset"  # Reemplaza con tu ruta
convert_all_yolo_to_masks(dataset_path)

