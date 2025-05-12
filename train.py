# train.py

from data_loader import load_dataset
from unet_model import unet_model
from utils import visualize_prediction

def main():
    print("Cargando datos...")
    X_train, X_val, y_train, y_val = load_dataset("Dataset/Training/Images", "Dataset/Training/Masks")

    print("Construyendo modelo...")
    model = unet_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Entrenando...")
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=20,
              batch_size=8)

    model.save("unet_periodontal.h5")
    print("Modelo guardado como unet_periodontal.h5")

    print("Visualizando predicción...")
    visualize_prediction(model, X_val, y_val, index=0)

if __name__ == "__main__":
    main()
