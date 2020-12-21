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


def default_file_filter(filename: str):
    return filename.endswith(".mp4")


default_transform = T.Compose(
    [
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class VideoDataset(Dataset):
    def __init__(self, path, metadata_filename="metadata.json", file_filter=None):
        self.path = path
        if file_filter is None:
            file_filter = default_file_filter
        self.file_filter = file_filter
        self.video_paths = [
            file for file in os.listdir(self.path) if self.file_filter(file)
        ]
        metadata = utils.load_json(os.path.join(self.path, metadata_filename))
        self.labels = {
            filename: LABEL_MAP[data["label"]] for filename, data in metadata.items()
        }

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        filename = self.video_paths[idx]
        label = self.labels[filename]
        video_full_path = os.path.join(self.path, filename)
        vframes, _, _ = read_video(
            video_full_path, pts_unit="sec", start_pts=0, end_pts=5
        )  # TODO fix hardcode
        vframes = vframes[::5]  # TODO fix hardcode
        # transforms
        vframes = vframes.permute(0, 3, 1, 2)
        vframes = vframes.type(torch.float64)
        n, c, _, _ = vframes.shape
        transformed_frames = torch.empty(n, c, IMAGE_SIZE, IMAGE_SIZE)
        for i, frame in enumerate(vframes):
            transformed_frames[i] = default_transform(frame)
        return transformed_frames, torch.tensor(label)


def get_dataset(data_path, n_workres):
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
    ds = VideoDataset(path=data_path)
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
