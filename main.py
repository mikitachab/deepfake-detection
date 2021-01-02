import os
import argparse

import torch
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from tqdm import tqdm

from deepfake_detection import (
    get_dataset,
    RCNN,
    SGDLearner,
    VideoDataset,
    VideoDatasetCV,
)


def argparse_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        default=os.path.join("data", "train_sample_videos"),
        type=str,
        dest="data_path",
    )
    parser.add_argument("--jobs", "-j", default=10, type=int)
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--use-old-cache", default=False, action="store_true")
    return parser


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = get_dataset(args.data_path, args.jobs, args.use_old_cache)
    model = RCNN()

    # learner = SGDLearner(model=model, dataset=dataset, device=device)
    # learner.fit(args.epochs)
    # print("score", learner.score_dataset())
    cv = VideoDatasetCV(KFold(n_splits=2))
    # scores = learner.cross_val_score(cv)
    scores = cross_val_score(cv=cv, model=model, dataset=dataset, device=device)
    print(scores)


def cross_val_score(cv, model, dataset, device):
    model_cls = model.__class__  # TODO make clone here (handle model params)
    scores = []
    for train_index, test_index in cv.split(dataset):
        print("make substests")
        train_ds = Subset(dataset, train_index)
        test_ds = Subset(dataset, test_index)
        model = model_cls().to(device)
        print("train")
        learner = SGDLearner(model=model, dataset=train_ds, device=device)
        learner.fit(1)
        print("test")
        score = learner.score(test_ds)
        scores.append(score)
    return scores


if __name__ == "__main__":
    parser = argparse_setup()
    args = parser.parse_args()
    main(args)
