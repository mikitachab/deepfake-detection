import os
import argparse

import torch
from sklearn.model_selection import KFold

from deepfake_detection import (
    get_dataset,
    RCNN,
    SGDLearner,
    VideoDatasetCV,
)
from deepfake_detection.cross_validation import cross_val_score


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

    learner = SGDLearner(model=model, dataset=dataset, device=device)
    learner.fit(args.epochs)
    print("score", learner.score_dataset())

    print("cv")
    cv = VideoDatasetCV(KFold(n_splits=2))
    scores = learner.cross_val_score(cv)
    scores = cross_val_score(cv=cv, model=model, dataset=dataset, device=device)
    print(scores)


if __name__ == "__main__":
    parser = argparse_setup()
    args = parser.parse_args()
    main(args)
