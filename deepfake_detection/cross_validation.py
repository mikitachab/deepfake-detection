import numpy as np
from torch.utils.data import Subset

from deepfake_detection import SGDLearner


class VideoDatasetCV:
    def __init__(self, cv):
        self.cv = cv

    def split(self, ds):
        y = ds.labels.numpy()
        fake_x = np.zeros_like(y)
        for train_index, test_index in self.cv.split(fake_x, y):
            yield train_index, test_index

    def get_n_splits(self):
        return self.cv.get_n_splits()


def cross_val_score(cv, model, dataset, device, epochs, score_device):
    scores = []
    n_splits = cv.get_n_splits()
    for i, (train_index, test_index) in enumerate(cv.split(dataset), 1):
        print("split {i}/{n_splits}".format(i=i, n_splits=n_splits))
        print("make substests")
        train_ds = Subset(dataset, train_index)
        test_ds = Subset(dataset, test_index)
        model = model.clone().to(device)

        print("train")
        # TODO how to setup? learner model should parameterized
        learner = SGDLearner(model=model, dataset=train_ds, device=device)
        result = learner.fit(epochs)

        print("test")
        score = learner.score(test_ds, device=score_device)

        scores.append({
            "train": result["train_scores"],
            "test": score
        })

        print("Train scores")
        for metric, score_val in result["train_scores"].items():
            print(metric, score_val)

        print("Test scores")
        for metric, score_val in score.items():
            print(metric, score_val)

    return scores
