import argparse
import os
import json

from tqdm import tqdm
import torch

from deepfake_detection.video_loader import Video2TensorLoader
from deepfake_detection.constants import LABEL_MAP, IMAGE_SIZE
from deepfake_detection.transforms import preprocessing_pipeline, default_transform

device = "cuda"

transforms_map = {
    "default": default_transform, 
    "preprocessing": preprocessing_pipeline(device=device)
}

PREPROCESSED_DATA_DIR = "preprocessed_data"
DATA_PATH = "data"


def get_meta():
    if os.path.exists("metadata.json"):
        with open("metadata.json") as f:
            return json.load(f)
    return {}

def main():
    metadata = get_meta()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs","-d", nargs="+")
    parser.add_argument("--transforms", "-t", type=str, choices=["default", "preprocessing"], default="preprocessing")
    args = parser.parse_args()

    transforms = transforms_map[args.transforms]
    loader = Video2TensorLoader(transforms=transforms)

    for dir_ in args.dirs:
        metadata_file = os.path.join(DATA_PATH, dir_, "metadata.json")

        with open(metadata_file) as file:
            data = json.load(file)
            metadata = {**metadata, **data}

        for file in tqdm(os.listdir(os.path.join(DATA_PATH, dir_))):
            video_file = os.path.join(DATA_PATH, dir_, file)
            if video_file.endswith(".mp4"):
                loader = Video2TensorLoader(transforms=transforms)
                t = loader.load(video_file)
                torch.save(t, os.path.join(PREPROCESSED_DATA_DIR, video_file.split("/")[-1]))
        
    with open("metadata.json", "w") as file:
        json.dump(metadata, file)

    print(len(metadata))

if __name__ == "__main__":
    main()
