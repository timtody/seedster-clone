import os
from attrs import frozen
import numpy as np
from skimage.transform import resize
from tensorflow.keras.models import load_model


@frozen
class Model:
    artifact = load_model(os.path.join("saved_models", "moisture.h5"))

    def predict(self, img):
        IMG_WIDTH = 2000
        IMG_HEIGHT = 2000
        img_n = img[
            int((img.shape[0] - IMG_WIDTH) / 2) : int(
                img.shape[0] - (img.shape[0] - IMG_WIDTH) / 2
            ),
            int((img.shape[1] - IMG_HEIGHT) / 2) : int(
                img.shape[1] - (img.shape[1] - IMG_HEIGHT) / 2
            ),
            :,
        ]
        img = np.expand_dims(
            resize(img_n, (224, 224), mode="constant", preserve_range=True), axis=-1
        )
        img = img[:, :, :, 0][np.newaxis, ...] / 255
        return self.artifact.predict(img)
