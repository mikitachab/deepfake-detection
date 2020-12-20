import numpy as np
import skimage as ski
from skimage import feature


class FaceExtract:
    def __init__(self, padding=10):
        self.padding = padding

    def __call__(self, image):
        trained_file = ski.data.lbp_frontal_face_cascade_filename()
        face_detector = feature.Cascade(trained_file)
        detected = face_detector.detect_multi_scale(
            img=image,
            scale_factor=1.2,
            step_ratio=1,
            min_size=(30, 30),
            max_size=(224, 224),
            min_neighbour_number=5,
        )
        if len(detected) == 0:
            return image
        (x, y, w, h) = patch_to_tuple(detected[0])
        p = self.padding
        cropped_face = image[y - p + 1 : y + h + p, x - p + 1 : x + w + p]
        if not validate_shape(cropped_face.shape):
            return image
        return cropped_face


class Resize:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, image):
        return ski.transform.resize(image, self.shape)


class ToArray:
    def __call__(self, image):
        return np.array(image)


class Sharp:
    def __call__(self, image):
        return ski.filters.unsharp_mask(image, multichannel=True)


class EqualizeHistogram:
    def __call__(self, image):
        return ski.exposure.equalize_hist(image)


class ToImage:
    def __call__(self, float_array):
        img = float_array.astype(np.float64) / float_array.max()
        img = 255 * img
        img = img.astype(np.uint8)
        return img


def patch_to_tuple(patch):
    return patch["c"], patch["r"], patch["width"], patch["height"]


def validate_shape(shape):
    return all(shape)
