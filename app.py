import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("GAN Image Enhancement Demo")

@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model("generator_model")

def preprocess_image(img):
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    noisy_img = img + np.random.normal(0, 0.1, img.shape)
    noisy_img = np.clip(noisy_img, 0, 1)
    return noisy_img[np.newaxis, ...]

def postprocess_image(img_tensor):
    img = img_tensor[0].numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

uploaded_file = st.file_uploader("Upload an Image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Original Image", use_column_width=True)

    noisy_img = preprocess_image(input_image)
    st.image(noisy_img[0], caption="Noisy Low-Res Input", use_column_width=True)

    model = load_model()
    enhanced_img = model.predict(noisy_img)
    enhanced_img_pil = postprocess_image(enhanced_img)

    st.image(enhanced_img_pil, caption="Enhanced Output by GAN", use_column_width=True)
