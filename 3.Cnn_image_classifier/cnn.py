import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io

# Load model
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model("mnist_cnn.h5")

model = load_model()

st.title("🧠 MNIST Digit Classifier")
st.markdown("Upload an image of a digit (0-9). The image can be handwritten or typed, but should be centered and clearly visible.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.convert("L")  # Grayscale
    image = ImageOps.invert(image)  # Invert background
    image = ImageOps.autocontrast(image)
    image = image.resize((28, 28))

    image_array = np.array(image) / 255.0
    image_array = 1.0 - image_array  # Convert to MNIST style

    # Reject image if it's likely blank
    if np.mean(image_array) < 0.1:
        return None

    return image_array.reshape(1, 28, 28, 1)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        processed = preprocess_image(image)

        if processed is None:
            st.error("❌ This does not appear to be a digit image.")
        else:
            prediction = model.predict(processed)
            confidence = np.max(prediction)
            predicted_class = np.argmax(prediction)

            if confidence < 0.5:
                st.error("❌ This does not appear to be a valid digit (0–9). Try uploading a clearer handwritten or printed digit.")
            else:
                st.success(f"✅ Predicted Digit: **{predicted_class}** with confidence **{confidence:.2f}**")
    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")
