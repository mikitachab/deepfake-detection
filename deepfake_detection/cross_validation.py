import numpy as np


class VideoDatasetCV:
    def __init__(self, cv):
        self.cv = cv

    def split(self, ds):
        y = ds.labels.numpy()
        fake_x = np.zeros_like(y)
        for train_index, test_index in self.cv.split(fake_x, y):
            yield train_index, test_index
