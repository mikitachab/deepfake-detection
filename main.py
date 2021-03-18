#!/usr/bin/env python3
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
from deepfake_detection.cnn import get_cnn
from deepfake_detection.cross_validation import cross_val_score

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    dataset = get_dataset(args)
    model = RCNN(cnn=get_cnn(args.cnn))

    if args.fit_and_score:
        print("performing fit and score for {} epochs".format(args.epochs))
        fit_and_score(model, dataset, args)

    if args.cv:
        print("cross val")
        cross_val(model, dataset)


def cross_val(model, dataset):
    cv = VideoDatasetCV(KFold(n_splits=2))
    scores = cross_val_score(cv, model, dataset, device)
    print(scores)


def fit_and_score(model, dataset, args):
    learner = SGDLearner(model=model, dataset=dataset, device=device)
    learner.fit(args.epochs)
    print("score", learner.score_dataset())

    if args.export:
        learner.export(args.export_path)


def argparse_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        default=os.path.join("data", "train_sample_videos"),
        type=str,
        dest="data_path",
    )
    parser.add_argument("--epochs", "-e", default=1, type=int,
        help="Number of epochs")
    parser.add_argument("--no-cache", action="store_true",
        help="not using data prom previous run")
    parser.add_argument("--fit-and-score", action="store_true",
        help="fit model and compute train score")
    parser.add_argument("--cv", action="store_true", 
        help="run cross validation")
    parser.add_argument(
        "--cnn", type=str, choices=["resnet18", "resnet34"], default="resnet18",
        help="Set used cnn"
    )
    parser.add_argument("--export-path", type=str, default="export.pth",
        help="filename where to save model")
    parser.add_argument("--export", action="store_true",
        help="set to save trained model")
    parser.add_argument("--no-preprocessing", action="store_true",
        help="not using preprocessing pipeline")

    return parser


if __name__ == "__main__":
    parser = argparse_setup()
    args = parser.parse_args()
    main(args)
