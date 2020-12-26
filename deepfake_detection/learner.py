import torch
from torch import nn
import torch.optim as optim

from sklearn.metrics import accuracy_score
from tqdm import tqdm


class SGDLearner:
    def __init__(self, model, dataset, loss_func=nn.CrossEntropyLoss(), device="cpu"):
        self.dataset = dataset
        self.device = device
        self.loss_func = loss_func
        self.model = model.to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def fit(self, epochs=1):
        self.model.train()
        for e in range(epochs):
            print("e =", e + 1)
            running_loss = 0.0
            with tqdm(total=len(self.dataset)) as pb:
                for i, (x, y) in enumerate(self.dataset):
                    x, y = x.to(self.device), y.to(self.device)
                    y = torch.unsqueeze(y, 0)
                    self.optimizer.zero_grad()
                    pred = self.model(x)
                    loss = self.loss_func(pred, y)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    pb.update(1)
                print("loss = ", loss.item())

    def predict(self, t):
        self.model.eval()
        return self.model(t)

    def score_dataset(self):
        """
        Note: due to memory allocation issue while evaluation,
        score compution on cpu
        """
        device = torch.device("cpu")
        model = self.model.to(device)
        model.eval()
        print("computing accuracy on dataset")
        y_true = []
        y_pred = []
        with tqdm(total=len(self.dataset)) as pb:
            for i, (x, y) in enumerate(self.dataset):
                x, y = x.to(device), y.to(device)
                y_true.append(y.item())
                pred = model(x)
                y_pred.append(torch.argmax(pred).item())
                pb.update(1)
        return accuracy_score(y_true, y_pred)
