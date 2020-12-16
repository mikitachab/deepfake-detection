import json

import cv2
import PIL
import numpy as np


def load_json(json_path):
    with open(json_path) as file:
        return json.load(file)


def pil_to_opencv(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def opencv_to_pil(opencv_image):
    return PIL.Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
