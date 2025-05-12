import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model, load_model
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.decomposition import PCA

IMG_SIZE = 256
model_path = 'unet_periodontal.h5'
image_path = 'Dataset/Test/Images/00929.jpg'

def get_model_with_intermediate_outputs():
    model = load_model(model_path)
    
    # Usar los nombres de capas que muestra el error
    layer_names = [
        'conv2d_1', 'max_pooling2d',
        'conv2d_3', 'max_pooling2d_1',
        'conv2d_5', 
        'up_sampling2d', 'conv2d_7',
        'up_sampling2d_1', 'conv2d_9',
        'conv2d_10'  # Capa de salida
    ]
    
    outputs = [model.get_layer(name).output for name in layer_names]
    return Model(inputs=model.input, outputs=outputs)

def visualize_process():
    try:
        # Cargar y preprocesar imagen
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Obtener modelo con salidas intermedias
        model = get_model_with_intermediate_outputs()
        outputs = model.predict(img_array)
        pred_mask = np.argmax(outputs[-1][0], axis=-1)

        # Configurar visualización
        plt.figure(figsize=(20, 12))
        
        # Imagen original
        plt.subplot(3, 4, 1)
        plt.imshow(img)
        plt.title("1. Imagen Original")
        plt.axis('off')

        # Capas intermedias
        for i, (output, name) in enumerate(zip(outputs[:-1], [
            '2. Encoder Conv1', '3. Encoder Pool1',
            '4. Encoder Conv2', '5. Encoder Pool2',
            '6. Middle Conv',
            '7. Decoder Upsample1', '8. Decoder Conv1',
            '9. Decoder Upsample2', '10. Decoder Conv2'
        ])):
            plt.subplot(3, 4, i+2)
            
            if len(output[0].shape) == 2:
                output = np.repeat(output[0][..., np.newaxis], 3, axis=-1)
                vis_img = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
            else:
                pca = PCA(n_components=3)
                flat_features = output[0].reshape(-1, output[0].shape[-1])
                reduced = pca.fit_transform(flat_features)
                vis_img = reduced.reshape(output[0].shape[0], output[0].shape[1], 3)
                vis_img = cv2.normalize(vis_img, None, 0, 255, cv2.NORM_MINMAX)
            
            plt.imshow(vis_img.astype(np.uint8))
            plt.title(name)
            plt.axis('off')

        # Resultados finales
        plt.subplot(3, 4, 11)
        plt.imshow(img)
        plt.imshow(pred_mask, cmap='jet', alpha=0.5)
        plt.title("11. Segmentación Superpuesta")
        plt.axis('off')

        plt.subplot(3, 4, 12)
        plt.imshow(pred_mask, cmap='jet')
        plt.title("12. Máscara de Segmentación")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Posibles soluciones:")
        print("1. Verifica que la imagen existe en la ruta especificada")
        print("2. Revisa que el modelo tenga la arquitectura esperada")
        print("3. Comprueba los nombres de las capas en model.summary()")

if __name__ == "__main__":
    visualize_process()