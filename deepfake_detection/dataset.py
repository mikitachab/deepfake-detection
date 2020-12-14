import torch
from torchvision import transforms as T
from torch.utils.data import Dataset

import torchvideo
from torchvideo.datasets import VideoFolderDataset
from torchvideo.samplers import ClipSampler
import torchvideo.transforms as VT

from deepfake_detection.constants import N_FRAMES, LABEL_MAP, IMAGE_SIZE
from deepfake_detection import utils


class WrapDataset(Dataset):
    def __init__(self, video_ds, transform=None):
        self.video_ds = video_ds
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

    def __len__(self):
        return len(self.video_ds)

    def __getitem__(self, idx):
        x, y = self.video_ds[idx]
        x = list(x)
        xts = torch.empty((len(x), 3, IMAGE_SIZE, IMAGE_SIZE))
        for i in range(len(x)):
            xts[i] = self.transform(x[i])
        return xts, torch.tensor(y)


def get_dataset():
    metadata = utils.load_json("data/train_sample_videos/metadata.json")
    labels = {name: LABEL_MAP[data["label"]] for name, data in metadata.items()}

    video_ds = VideoFolderDataset(
        root_path="data/train_sample_videos/",
        sampler=ClipSampler(clip_length=N_FRAMES),
        label_set=labels,
        transform=VT.IdentityTransform(),
    )

    default_transform = T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = WrapDataset(video_ds, transform=default_transform)
    return dataset
