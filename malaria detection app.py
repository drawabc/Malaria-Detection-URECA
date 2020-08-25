import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from load_model import create_model
from keras.preprocessing.image import load_img, img_to_array


def predict(image):
    model = create_model()
    #image = load_img(, target_size=(100, 100))
    image = ImageOps.fit(image, (100, 100))
    image = img_to_array(image)
    image = image.reshape((1, 100, 100, 3))
    yhat = model.predict(image)
    label = yhat[0][0]
    # return highest probability
    return label

### Excluding Imports ###
st.title("Malaria Detection")

uploaded_file = st.file_uploader("Upload Cell image...", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=200)
    st.write("")
    st.write("Classifying...")
    label = predict(image)
    if label == 0:
        st.write("Parasitized")
    else:
        st.write('Uninfected')




