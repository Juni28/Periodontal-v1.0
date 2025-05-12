# unet_model.py

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES

def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    inputs = Input(input_size)
    
    # Downsampling
    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    # Bottleneck
    c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(128, 3, activation='relu', padding='same')(c3)

    # Upsampling
    u4 = UpSampling2D()(c3)
    m4 = concatenate([u4, c2])
    c4 = Conv2D(64, 3, activation='relu', padding='same')(m4)
    c4 = Conv2D(64, 3, activation='relu', padding='same')(c4)

    u5 = UpSampling2D()(c4)
    m5 = concatenate([u5, c1])
    c5 = Conv2D(32, 3, activation='relu', padding='same')(m5)
    c5 = Conv2D(32, 3, activation='relu', padding='same')(c5)

    outputs = Conv2D(num_classes, 1, activation='softmax')(c5)

    return Model(inputs=[inputs], outputs=[outputs])

