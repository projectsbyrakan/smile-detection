import streamlit as st
import requests
import base64
import os
from io import BytesIO
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Smile Detection", layout="centered")
st.title("Smile Detection")

model = st.selectbox("Choose Model", ["efficientnet", "mobilenet"])
mode = st.radio("Input Type", ["Image", "Video"])

if mode == "Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        st.image(uploaded, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Smiles"):
            with st.spinner("Processing..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    params={"model": model}
                )

            if response.status_code == 200:
                data = response.json()
                img_bytes = base64.b64decode(data["annotated_image"])
                result_img = Image.open(BytesIO(img_bytes))
                st.image(result_img, caption="Result", use_container_width=True)

                if data["faces_detected"] == 0:
                    st.warning("No faces detected in the image.")
                else:
                    st.success(f"Faces detected: {data['faces_detected']}")
                    for i, face in enumerate(data["results"], 1):
                        st.write(f"**Face {i}:** {face['label']} ({face['confidence']:.1%})")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

elif mode == "Video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded:
        st.video(uploaded)

        if st.button("Detect Smiles"):
            with st.spinner("Processing video... This may take a while."):
                response = requests.post(
                    f"{API_URL}/predict-video",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    params={"model": model},
                    timeout=300
                )

            if response.status_code == 200:
                data = response.json()
                video_bytes = base64.b64decode(data["annotated_video"])
                st.video(video_bytes)
                st.success(f"Frames processed: {data['frames_processed']} | Total faces detected: {data['total_faces_detected']}")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")