import torch
import torchvision.transforms as T
import numpy as np
from facenet_pytorch import MTCNN

def patch_to_tuple(patch):
    return patch["c"], patch["r"], patch["width"], patch["height"]


def validate_shape(shape):
    return all(shape)


class UnsharpMask:
    def __init__(self, amount=1, low_pass_filter=None, device=torch.device("cpu")):
        self.device = device
        self.amount = amount
        if low_pass_filter is None:
            low_pass_filter = T.GaussianBlur(3)
        self.low_pass_filter = low_pass_filter

    def __call__(self, image):
        image = image.to(self.device)
        blurred = self.low_pass_filter(image)
        mask = image - blurred
        return image + self.amount * mask


class ToImage:
    def __call__(self, float_image):
        img = float_image.astype(np.float64) / float_image.max()
        img = 255 * img
        img = img.astype(np.uint8)
        return img


# TODO resolve strange result
class EqualizeHistogram:
    def __init__(self, device=torch.device("cpu")):
        self.device = device

    def __call__(self, image):
        image = image.to(self.device)
        result_img = torch.empty_like(image).to(self.device)
        result_img[0] = self._scale_single_channel(image[0])
        result_img[1] = self._scale_single_channel(image[1])
        result_img[2] = self._scale_single_channel(image[2])
        return result_img

    # original comes from https://github.com/pytorch/vision/issues/1049
    def _scale_single_channel(self, im):
        """Scale the data in the channel to implement equalize."""
        # Compute the histogram of the image channel.
        # TODO resolve issue with negative values in extracted face
        im = im.type(torch.uint8).float()
        histo = torch.histc(im, bins=256, min=0, max=255)  # .type(torch.int32)
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero_histo = torch.reshape(histo[histo != 0], [-1])
        step = (torch.sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (torch.cumsum(histo, 0) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = torch.cat([torch.zeros(1).to(self.device), lut[:-1]])
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return torch.clamp(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        if step == 0:
            result = im
        else:
            # can't index using 2d index. Have to flatten and then reshape
            lut = build_lut(histo, step)
            # print(torch.unique(im.flatten().long()))
            result = torch.gather(lut, 0, im.flatten().long())
            result = result.reshape_as(im)

        return result


class FaceExtractMTCNN:
    def __init__(self, device=torch.device("cpu")):
        self.device = device
        self.mtcnn = MTCNN(post_process=False, device=self.device)

    def __call__(self, image):
        mtcnn_input = torch.unsqueeze(image.permute(1, 2, 0), 0)
        try:
            faces = self.mtcnn(mtcnn_input)
        except TypeError:
            return image
        if len(faces) == 0:
            return image
        return faces[0]
