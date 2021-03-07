import io
import os
import contextlib
import tempfile

import torch
from torchvision import transforms as T
import streamlit as st

from deepfake_detection.video_loader import Video2TensorLoader
from deepfake_detection.constants import LABEL_MAP
from deepfake_detection.preprocessing import (
    FaceExtractMTCNN,
    EqualizeHistogram,
    UnsharpMask,
)

IMAGE_SIZE = 224
device = "cpu"

st.set_option("deprecation.showfileUploaderEncoding", False)

default_transform = T.Compose(
    [
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


transforms = T.Compose(
    [
        FaceExtractMTCNN(device=device),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        UnsharpMask(device=device),
        EqualizeHistogram(device=device),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.load("export.pth")
    model.eval()
    return model


@contextlib.contextmanager
def tempfile_with_content(content):
    fd, path = tempfile.mkstemp()
    os.close(fd)
    with open(path, "wb") as f:
        f.write(content)
    yield path
    os.remove(path)


def extract_prediction(pred_tensor):
    print(pred_tensor)
    index = torch.argmax(pred_tensor).item()
    label = {v: k for k, v in LABEL_MAP.items()}[index]
    conf = pred_tensor[0][index].item()
    return label, conf


def main():
    model = load_model().to("cpu")

    st.title("DeepFake detection")

    uploaded_file = st.file_uploader("Select video to upload")
    if uploaded_file:
        with tempfile_with_content(uploaded_file.getvalue()) as file:
            with st.spinner("file"):
                loader = Video2TensorLoader(transforms=transforms)
                input_tensor = loader.load(file)

            with st.spinner("getting prediction"):
                result = model(input_tensor)

            label, conf = extract_prediction(result)

            if label == "FAKE":
                st.error(f"FAKE ({conf:.2f})")
            else:
                st.success(f"REAL ({conf:.2f})")


if __name__ == "__main__":
    main()
