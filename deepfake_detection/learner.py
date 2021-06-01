import torch
from torch import nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm


class SGDLearner:
    def __init__(
        self,
        model,
        dataset,
        loss_func=nn.CrossEntropyLoss(),
        device="cpu",
        data_limit=None,
    ):
        self.dataset = dataset
        self.device = device
        self.loss_func = loss_func
        self.model = model.to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def fit(self, epochs=1):
        self.model.train()
        epochs_loss = []
        for e in range(epochs):
            print("e =", e + 1)
            running_loss = 0.0
            with tqdm(total=len(self.dataset), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pb:
                ys = []
                preds = []
                for i, (x, y) in enumerate(self.dataset):
                    x, y = x.to(self.device), y.to(self.device)
                    y = torch.unsqueeze(y, 0)
                    self.optimizer.zero_grad()
                    pred = self.model(x)

                    loss = self.loss_func(pred, y)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()

                    ys.append(y.item())
                    preds.append(torch.argmax(pred).item())

                    pb.update(1)
                print("loss = ", running_loss)
            epochs_loss.append(running_loss)

        return {
            "loss": epochs_loss,
            "train_scores": make_scores(ys, preds)
        }

    def predict(self, t):
        self.model.eval()
        return self.model(t)

    def score_dataset(self):
        return self.score(self.dataset)

    def score(self, dataset, device):
        """
        Note: due to memory allocation issue while evaluation,
        score compution on cpu
        """
        model = self.model.to(device)
        model.eval()
        y_true = []
        y_pred = []
        print("SGDLearner: computing accuracy on dataset")
        with tqdm(total=len(dataset), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pb:
            for x, y in dataset:
                x, y = x.to(device), y.to(device)
                y_true.append(y.item())
                pred = model(x)
                y_pred.append(torch.argmax(pred).item())
                pb.update(1)
        return make_scores(y_true, y_pred)

    def export(self, path):
        torch.save(self.model, path)


def make_scores(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }
