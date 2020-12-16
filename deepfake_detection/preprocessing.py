import numpy as np

import cv2
from skimage.filters import unsharp_mask
from skimage import exposure

from deepfake_detection import utils


class FaceExtract:
    def __init__(self, cascade_model="haarcascade_frontalface_default.xml", padding=10):
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + cascade_model
        )
        self.padding = padding

    def __call__(self, image):
        image = utils.pil_to_opencv(image)
        faces_detected = self.face_detector.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=5
        )
        if len(faces_detected) == 0:
            return image
        (x, y, w, h) = faces_detected[0]
        p = self.padding
        cropped_face = image[y - p + 1 : y + h + p, x - p + 1 : x + w + p]
        return utils.opencv_to_pil(cropped_face)


class ToArray:
    def __call__(self, image):
        return np.array(image)


class Sharp:
    def __call__(self, image):
        return unsharp_mask(image, multichannel=True)


class EqualizeHistogram:
    def __call__(self, image):
        return exposure.equalize_hist(image)


class ToImage:
    def __call__(self, float_array):
        img = float_array.astype(np.float64) / float_array.max()
        img = 255 * img
        img = img.astype(np.uint8)
        return img
