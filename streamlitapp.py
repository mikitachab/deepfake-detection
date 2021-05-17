import io
import os
import contextlib
import tempfile

import torch
import streamlit as st

from deepfake_detection.video_loader import Video2TensorLoader
from deepfake_detection.constants import LABEL_MAP, IMAGE_SIZE
from deepfake_detection.transforms import preprocessing_pipeline, default_transform


st.set_option("deprecation.showfileUploaderEncoding", False)
device = "cpu"

transforms_map = {
    "default": default_transform, 
    "preprocessing pipeline": preprocessing_pipeline(device=device)
}

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

    selected_tfms = st.selectbox("Preprocessing type", list(transforms_map.keys()))
    transforms = transforms_map[selected_tfms]

    uploaded_file = st.file_uploader("Select video to upload")
    if uploaded_file:
        with tempfile_with_content(uploaded_file.getvalue()) as file:
            with st.spinner("preprocessing"):
                loader = Video2TensorLoader(transforms=transforms)
                input_tensor = loader.load(file)

            with st.spinner("predicting"):
                result = model(input_tensor)

            label, conf = extract_prediction(result)

            if label == "FAKE":
                st.error(f"FAKE ({conf:.2f})")
            else:
                st.success(f"REAL ({conf:.2f})")


if __name__ == "__main__":
    main()
