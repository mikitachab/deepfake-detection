import os

import torch
from torchvision.io import read_video

from deepfake_detection.constants import IMAGE_SIZE


class Video2TensorLoader:
    def __init__(self, base_path=".", transforms=None, frame_size=IMAGE_SIZE):
        self.base_path = base_path
        if transforms is None:
            transforms = lambda x: x
        self.transforms = transforms

    def load(self, filename):
        vframes = self._get_frames_tensor(filename)
        n, c, _, _ = vframes.shape
        transformed_frames = torch.empty(n, c, IMAGE_SIZE, IMAGE_SIZE)
        for i, frame in enumerate(vframes):
            transformed_frames[i] = self.transforms(frame)
        return transformed_frames

    def _get_frames_tensor(self, filename):
        video_full_path = os.path.join(self.base_path, filename)
        vframes, _, _ = read_video(
            video_full_path, pts_unit="sec", start_pts=0, end_pts=5
        )  # TODO fix hardcode
        vframes = vframes[::5]  # TODO fix hardcode, add clipper
        vframes = vframes.permute(0, 3, 1, 2)
        vframes = vframes.type(torch.float64)
        return vframes
