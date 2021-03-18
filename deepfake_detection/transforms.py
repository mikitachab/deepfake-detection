from torchvision import transforms as T

from deepfake_detection.preprocessing import (
    FaceExtractMTCNN,
    EqualizeHistogram,
    UnsharpMask,
)
from deepfake_detection.constants import IMAGE_SIZE

default_transform = T.Compose(
    [
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocessing_pipeline(device):
    return T.Compose(
        [
            FaceExtractMTCNN(device=device),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            UnsharpMask(device=device),
            EqualizeHistogram(device=device),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
 