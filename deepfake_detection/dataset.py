import os
import shutil

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_video

from deepfake_detection.constants import LABEL_MAP, IMAGE_SIZE
from deepfake_detection import utils
from deepfake_detection.preprocessing import (
    FaceExtractMTCNN,
    EqualizeHistogram,
    UnsharpMask,
)
from deepfake_detection.video_loader import Video2TensorLoader
from deepfake_detection.transforms import preprocessing_pipeline, default_transform


class VideoDataset(Dataset):
    default_transform = T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(
        self,
        path,
        no_cache,
        transforms=None,
        metadata_filename="metadata.json",
        file_filter=None,
    ):
        self.path = path
        if file_filter is None:
            file_filter = _default_file_filter
        self.file_filter = file_filter
        if transforms is None:
            transforms = self.default_transform
        self.transforms = transforms
        self.video_paths = self._get_video_paths()
        self.labels_map = self._load_labels(metadata_filename)
        self.video_loader = Video2TensorLoader(self.path, self.transforms)
        self.cache = VideoDataCache("data/cache", no_cache)  # TODO fix hardcode

    def _load_labels(self, metadata_filename):
        metadata = utils.load_json(os.path.join(self.path, metadata_filename))
        return {
            filename: LABEL_MAP[data["label"]] for filename, data in metadata.items()
        }

    def _get_video_paths(self):
        return [file for file in os.listdir(self.path) if self.file_filter(file)]

    def __len__(self):
        return len(self.video_paths)

    def _get_frames_tensor(self, filename):
        video_full_path = os.path.join(self.path, filename)
        vframes, _, _ = read_video(
            video_full_path, pts_unit="sec", start_pts=0, end_pts=5
        )  # TODO fix hardcode
        vframes = vframes[::5]  # TODO fix hardcode, add clipper
        vframes = vframes.permute(0, 3, 1, 2)
        vframes = vframes.type(torch.float64)
        return vframes

    def __getitem__(self, idx):
        filename = self.video_paths[idx]
        label = self.labels_map[filename]
        if self.cache.cached.get(filename):
            vframes = self.cache.get(filename)
            return vframes, torch.tensor(label)

        transformed_frames = self.video_loader.load(filename)

        self.cache.save(filename, transformed_frames)
        return transformed_frames, torch.tensor(label)

    @property
    def labels(self):
        return torch.tensor([self.labels_map[path] for path in self.video_paths])


def _default_file_filter(filename):
    return filename.endswith(".mp4")


class VideoDataCache:
    def __init__(self, cache_path, no_cache):
        self.cache_path = cache_path
        self.cached = {}
        if no_cache:
            shutil.rmtree(self.cache_path, ignore_errors=True)
            os.makedirs(self.cache_path, exist_ok=True)
            print("cached cleared")
        else:
            self.cached = self._get_prepopulated_cached()

    def _get_prepopulated_cached(self):
        return {filename: True for filename in os.listdir(self.cache_path)}

    def get(self, filename):
        if self.cached.get(filename):
            full_path = os.path.join(self.cache_path, filename)
            return torch.load(full_path)
        return None

    def save(self, filename, tensor):
        if not self.cached.get(filename):
            full_path = os.path.join(self.cache_path, filename)
            torch.save(tensor, full_path)
            self.cached[filename] = True


def get_dataset(args):
    print("reading data from: ", args.data_path)
    print("creating video dataset")
    device = torch.device("cuda")

    transforms = (
        default_transform if args.no_preprocessing else preprocessing_pipeline(device)
    )

    ds = VideoDataset(
        path=args.data_path, no_cache=args.no_cache, transforms=transforms
    )
    return ds
