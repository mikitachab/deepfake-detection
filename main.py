import torch

from deepfake_detection import get_dataset, RCNN, SGDLearner


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = RCNN().to(device)
    dataset = get_dataset("data/train_sample_videos")

    learner = SGDLearner(model=model, dataset=dataset, device=device)
    learner.fit(1)


if __name__ == "__main__":
    main()
