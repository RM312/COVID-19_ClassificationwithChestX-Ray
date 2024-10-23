import streamlit as st
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('model.h5')

# Streamlit App
st.title("COVID-19 Chest X-ray Predictor")
st.write("Upload a chest X-ray image to get a COVID-19 prediction.")

# Upload an image file
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Chest X-ray Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the uploaded image to match the input format for the model
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image to [0,1]

    # Make the prediction
    prediction = model.predict(img_array)

    # Interpret the prediction
    if prediction > 0.6890:
        st.write("The report is **COVID-19 Negative**")
    else:
        st.write("The report is **COVID-19 Positive**")

    # Plot the image
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    st.pyplot(plt)
