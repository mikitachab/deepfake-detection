import os
import concurrent.futures
import multiprocessing as mp

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_video

from deepfake_detection.constants import N_FRAMES, LABEL_MAP, IMAGE_SIZE
from deepfake_detection import utils
from deepfake_detection.preprocessing import (
    ToArray,
    Resize,
    FaceExtract,
    Sharp,
    EqualizeHistogram,
    ToImage,
)


class VideoDataCache:
    def __init__(self, cache_path, use_old):
        self.cache_path = cache_path
        self.cached = {}
        if use_old:
            self.cached = self._get_prepopulated_cached()
        else:
            os.makedirs(self.cache_path, exist_ok=True)

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


def default_file_filter(filename: str):
    return filename.endswith(".mp4")


default_transform = T.Compose(
    [
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class VideoDataset(Dataset):
    def __init__(
        self, path, use_old_cache, metadata_filename="metadata.json", file_filter=None
    ):
        self.path = path
        if file_filter is None:
            file_filter = default_file_filter
        self.file_filter = file_filter
        self.video_paths = self._get_video_paths()
        self.labels = self._load_labels(metadata_filename)
        self.cache = VideoDataCache("data/cache", use_old_cache)  # TODO fix hardcode

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
        label = self.labels[filename]
        if self.cache.cached.get(filename):
            vframes = self.cache.get(filename)
            return vframes, torch.tensor(label)
        vframes = self._get_frames_tensor(filename)
        n, c, _, _ = vframes.shape
        transformed_frames = torch.empty(n, c, IMAGE_SIZE, IMAGE_SIZE)
        for i, frame in enumerate(vframes):
            transformed_frames[i] = default_transform(frame)
        self.cache.save(filename, transformed_frames)
        return transformed_frames, torch.tensor(label)


def get_dataset(data_path, n_workres, use_old_cache):
    print("reading data from: ", data_path)
    # metadata_path = os.path.join(data_path, "metadata.json")
    # metadata = utils.load_json(metadata_path)
    # labels = {name: LABEL_MAP[data["label"]] for name, data in metadata.items()}

    print("creating video dataset")
    # video_ds = VideoFolderDataset(
    #     root_path=data_path,
    #     sampler=ClipSampler(clip_length=N_FRAMES),
    #     label_set=labels,
    #     transform=VT.IdentityTransform(),
    # )

    # default_transform = T.Compose(
    #     [
    #         ToArray(),
    #         Resize((224, 224)),
    #         FaceExtract(),  # pickle error
    #         Sharp(),
    #         EqualizeHistogram(),
    #         ToImage(),
    #         T.ToTensor(),
    #         T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    # dataset = WrapDataset(video_ds, transform=default_transform, n_workres=n_workres)
    # return dataset
    ds = VideoDataset(path=data_path, use_old_cache=use_old_cache)
    return ds


# class WrapDataset(Dataset):
#     def __init__(self, video_ds, n_workres, transform=None):
#         self.video_ds = video_ds
#         self.n_workres = n_workres
#         if transform is None:
#             self.transform = lambda x: x
#         else:
#             self.transform = transform

#     def __len__(self):
#         return len(self.video_ds)

#     def __getitem__(self, idx):
#         x, y = self.video_ds[idx]
#         with mp.get_context("spawn").Pool(self.n_workres) as pool:
#             processed_x = pool.map(self.transform, list(x))
#         xts = torch.empty((len(processed_x), 3, IMAGE_SIZE, IMAGE_SIZE))
#         for i, timage in enumerate(processed_x):
#             xts[i] = processed_x[i]
#         return xts, torch.tensor(y)
