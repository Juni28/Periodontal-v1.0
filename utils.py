# utils.py

import numpy as np
import matplotlib.pyplot as plt

def visualize_prediction(model, X_val, y_val, index=0):
    pred = model.predict(np.expand_dims(X_val[index], axis=0))[0]
    pred_mask = np.argmax(pred, axis=-1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(X_val[index])
    plt.title("Imagen")
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.argmax(y_val[index], axis=-1))
    plt.title("Máscara Real")
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask)
    plt.title("Predicción")
    plt.show()
