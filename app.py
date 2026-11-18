
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾"
)

st.title("🐾 Cat vs Dog Image Classifier")
st.write("Upload an image to identify whether it contains a Cat or Dog.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "final_cat_dog_model.keras"
    )

model = load_model()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    image = image.resize((128, 128))

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    probability = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    if probability >= 0.5:
        prediction = "Dog"
        confidence = probability * 100
    else:
        prediction = "Cat"
        confidence = (1 - probability) * 100

    st.subheader("Prediction")
    st.success(f"Predicted Class: {prediction}")

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    st.write(f"Cat probability: {(1-probability)*100:.2f}%")
    st.write(f"Dog probability: {probability*100:.2f}%")
