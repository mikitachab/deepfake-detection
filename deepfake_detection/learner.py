import torch
from torch import nn
import torch.optim as optim

from tqdm import tqdm


class SGDLearner:
    def __init__(self, model, dataset, loss_func=nn.CrossEntropyLoss(), device="cpu"):
        self.dataset = dataset
        self.device = device
        self.loss_func = loss_func
        self.model = model.to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def fit(self, epochs=1):
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
