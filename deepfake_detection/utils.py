import json

import cv2
import PIL
import numpy as np


def load_json(json_path):
    with open(json_path) as file:
        return json.load(file)
