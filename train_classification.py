# entrenar_clasificador.py

import os
import cv2
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf

# CONFIG
IMG_HEIGHT, IMG_WIDTH = 256, 256
IMAGE_DIR = "Dataset/Training/Images"
MASK_DIR = "Dataset/Training/Masks"
CSV_PATH = "Dataset/Training/Train_captions.csv"

# FUNCIONES
def extraer_etiqueta(texto):
    niveles = [int(n) for n in re.findall(r'level of gingivitis (\d+)', texto)]
    if not niveles:
        return "Sano"
    max_nivel = max(niveles)
    if max_nivel == 1:
        return "Gingivitis leve"
    elif max_nivel in [2, 3]:
        return "Gingivitis moderada"
    else:
        return "Periodontitis severa"

def cargar_datos():
    df = pd.read_csv(CSV_PATH)
    df['Etiqueta'] = df['BS1'].apply(extraer_etiqueta)
    df_simple = df.groupby("FileName").first().reset_index()

    le = LabelEncoder()
    df_simple['label'] = le.fit_transform(df_simple['Etiqueta'])

    X, y = [], []

    for _, row in df_simple.iterrows():
        try:
            img_path = os.path.join(IMAGE_DIR, row['FileName'])
            mask_path = os.path.join(MASK_DIR, row['FileName'].replace('.jpg', '.png'))

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
            mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
            mask = np.expand_dims(mask, axis=-1)

            combinado = np.concatenate((img, mask), axis=-1)
            X.append(combinado)
            y.append(row['label'])
        except:
            print(f"Error al procesar: {row['FileName']}")

    X = np.array(X)
    y = to_categorical(np.array(y))
    return X, y, le

# CARGAR DATOS
X, y, label_encoder = cargar_datos()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# MODELO
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 4)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# GUARDAR MODELO
model.save("modelo_clasificacion_diagnostico.h5")
print("Modelo guardado como modelo_clasificacion_diagnostico.h5")
print("Clases:", label_encoder.classes_)
 