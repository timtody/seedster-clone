import os

import cv2
from attrs import frozen
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize
from keras.models import load_model as load_keras_model


@frozen
class Model:
    artifact_segmentation = load_keras_model(os.path.join("saved_models", "SeedSeg.h5"))
    artifact_tkw = load_model(os.path.join("saved_models", "tkw.h5"))

    def segmentation(self, img):
        IMG_HEIGHT = 1024
        IMG_WIDTH = 1024
        img = img[
            int((img.shape[0] - 2000) / 2) : int(
                img.shape[0] - (img.shape[0] - 2000) / 2
            ),
            int((img.shape[1] - 2000) / 2) : int(
                img.shape[1] - (img.shape[1] - 2000) / 2
            ),
            :,
        ]
        img_n = np.expand_dims(
            cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA),
            axis=-1,
        )
        preds_test = self.artifact_segmentation.predict(np.expand_dims(img_n, axis=0), verbose=1)
        predicted_img = np.reshape(preds_test, (IMG_HEIGHT, IMG_WIDTH))
        nrn = resize(
            np.squeeze(predicted_img),
            (img.shape[0], img.shape[1]),
            mode="constant",
            preserve_range=True,
        )
        nrn = (nrn > 0.70).astype(np.uint8)
        mask = nrn * 255
        img[np.where(mask == 255)] = 225
        return cv2.applyColorMap(img[:, :, 0], cv2.COLORMAP_JET)

    def predict_tkw(self, img):
        img_width = 224
        img_height = 224
        img = np.expand_dims(
            resize(img, (img_height, img_width), mode="constant", preserve_range=True),
            axis=-1,
        )
        X_Im = img[:, :, :, 0][np.newaxis, ...] / 255
        y = self.artifact_tkw.predict(X_Im)
        return np.around(y, 3)

    def predict(self, img):
        return self.predict_tkw(self.segmentation(img))
