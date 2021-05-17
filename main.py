#!/usr/bin/env python3
import os
import argparse
import random

import torch
from sklearn.model_selection import KFold
import requests

from deepfake_detection import (
    get_dataset,
    RCNN,
    SGDLearner,
    VideoDatasetCV,
)
from deepfake_detection.cross_validation import cross_val_score

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def main(args):
    random.seed(1410)
    torch.manual_seed(1410)

    if args.cpu:
        global device
        device = "cpu"

    dataset = get_dataset(args)
    model = RCNN(cnn=args.cnn, rnn_hidden_size=args.rnn_hidden_size, rnn_num_layers=args.rnn_num_layers)

    print("CNN: ", args.cnn)

    if args.fit_and_score:
        print("performing fit and score for {} epochs".format(args.epochs))
        fit_and_score(model, dataset, args)

    if args.cv:
        print("cross val")
        scores = cross_val(model, dataset, epochs=args.epochs)

        if args.send_cv:
            send_cv(args, scores)

def send_cv(args, scores):
    preprocessing = "preprocessing_pipeline"
    if args.no_preprocessing:
        preprocessing = "no_preprocessing"

    data = {
        "preprocessing": preprocessing,
        "cnn": args.cnn,
        "splits": scores,
        "description": args.desc
    }

    print("sending results")
    print(data)

    r = requests.post(args.db_url, json=data, headers={
        "X-RESULTS-SECRET": os.getenv("RESULTS_SECRET")
    })

    print("response status", r.status_code)
    print("response data", r.json())


def cross_val(model, dataset,epochs):
    cv = VideoDatasetCV(KFold(n_splits=5, shuffle=True, random_state=1410))
    scores = cross_val_score(cv, model, dataset, device, epochs)
    print(scores) # TODO save score to file instead of printing
    return scores


def fit_and_score(model, dataset, args):
    learner = SGDLearner(model=model, dataset=dataset, device=device)
    s = learner.fit(args.epochs)
    # print("score", learner.score_dataset())
    print(s)

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
        "--cnn", type=str, choices=["resnet18", "resnet34", "resnet50", "vgg11", "vgg13", "vgg16"], default="resnet18",
        help="Set used cnn"
    )
    parser.add_argument("--export-path", type=str, default="export.pth",
        help="filename where to save model")
    parser.add_argument("--export", action="store_true",
        help="set to save trained model")
    parser.add_argument("--no-preprocessing", action="store_true",
        help="not using preprocessing pipeline")
    parser.add_argument("--cache-dir", "-c", type=str,
        help="cache directory")
    parser.add_argument("--rnn-hidden-size", type=int, default=128)
    parser.add_argument("--rnn-num-layers", type=int, default=2)
    parser.add_argument("--data-limit", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--send-cv", action="store_true")
    parser.add_argument("--db-url", type=str)
    parser.add_argument("--desc", type=str, default="")

    return parser


if __name__ == "__main__":
    parser = argparse_setup()
    args = parser.parse_args()
    main(args)
