import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from deepfake_detection import get_dataset, RCNN


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = RCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    dataset = get_dataset()

    for e in range(10):
        print("e=", e + 1)
        running_loss = 0.0
        with tqdm(total=len(dataset)) as pb:
            for i, (x, y) in enumerate(dataset):
                x, y = x.to(device), y.to(device)
                y = torch.unsqueeze(y, 0)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pb.update(1)
        print("loss", loss.item())


if __name__ == "__main__":
    main()
