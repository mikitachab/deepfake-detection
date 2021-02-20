import torchvision


def get_cnn(model_name="resnet18"):
    model_getter = getattr(torchvision.models, model_name)
    return model_getter(pretrained=True)
