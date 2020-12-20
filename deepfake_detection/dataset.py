import os
import concurrent.futures
import multiprocessing as mp

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset

import torchvideo
from torchvideo.datasets import VideoFolderDataset
from torchvideo.samplers import ClipSampler
import torchvideo.transforms as VT

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


class WrapDataset(Dataset):
    def __init__(self, video_ds, n_workres, transform=None):
        self.video_ds = video_ds
        self.n_workres = n_workres
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

    def __len__(self):
        return len(self.video_ds)

    def __getitem__(self, idx):
        x, y = self.video_ds[idx]
        with mp.get_context("spawn").Pool(self.n_workres) as pool:
            processed_x = pool.map(self.transform, list(x))
        xts = torch.empty((len(processed_x), 3, IMAGE_SIZE, IMAGE_SIZE))
        for i, timage in enumerate(processed_x):
            xts[i] = processed_x[i]
        return xts, torch.tensor(y)


def get_dataset(data_path, n_workres):
    print("reading data from: ", data_path)
    metadata_path = os.path.join(data_path, "metadata.json")
    metadata = utils.load_json(metadata_path)
    labels = {name: LABEL_MAP[data["label"]] for name, data in metadata.items()}

    print("creating video dataset")
    video_ds = VideoFolderDataset(
        root_path=data_path,
        sampler=ClipSampler(clip_length=N_FRAMES),
        label_set=labels,
        transform=VT.IdentityTransform(),
    )

    default_transform = T.Compose(
        [
            ToArray(),
            Resize((224, 224)),
            FaceExtract(),  # pickle error
            Sharp(),
            EqualizeHistogram(),
            ToImage(),
            T.ToTensor(),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = WrapDataset(video_ds, transform=default_transform, n_workres=n_workres)
    return dataset
